import { useCallback, useEffect, useState } from "react";
import { ApiError, DEFAULT_API_BASE, api, getApiBase, setApiBase } from "../../lib/api";
import { duration, fmtScore, humanize } from "../../lib/format";
import type { Evaluation, FeedbackKind, Preferences, RecommendationItem, SearchItem, VideoCard } from "../../lib/types";
import { QualityBreakdown } from "./QualityBreakdown";
import { AssessmentBadge, Empty, ErrorBox, QualityPill, Spinner } from "./common";

type Select = (id: string) => void;

function VideoMeta({ v }: { v: VideoCard }) {
  return (
    <div className="muted small">
      {[v.channel_title, v.genre ? humanize(v.genre) : null, duration(v.duration_seconds), v.language?.toUpperCase()]
        .filter(Boolean)
        .join(" · ")}
      {v.is_synthetic ? <span className="tag">synthetic demo</span> : null}
    </div>
  );
}

// ------------------------------------------------------------------------------------- Discover

function RecCard({ item, onSelect }: { item: RecommendationItem; onSelect: Select }) {
  const [sent, setSent] = useState<FeedbackKind | null>(null);
  const [why, setWhy] = useState(false);
  const send = (kind: FeedbackKind) =>
    api
      .feedback({ video_id: item.video.id, kind, impression_id: item.impression_id })
      .then(() => setSent(kind))
      .catch(() => setSent(null));
  return (
    <li className="card">
      <div className="card-head">
        <button type="button" className="title" onClick={() => onSelect(item.video.id)}>
          {item.video.title}
        </button>
        <QualityPill q={item.intrinsic_quality} />
      </div>
      <VideoMeta v={item.video} />
      {item.ranking.exploration ? <span className="tag">exploration</span> : null}
      <ul className="why">
        {item.explanations.slice(0, why ? undefined : 2).map((e) => (
          <li key={e}>{e}</li>
        ))}
      </ul>
      {why ? (
        <table className="contrib">
          <tbody>
            {Object.entries(item.ranking.contributions)
              .filter(([, v]) => v !== 0)
              .sort((a, b) => b[1] - a[1])
              .map(([k, v]) => (
                <tr key={k}>
                  <td>{humanize(k)}</td>
                  <td className="num">{item.ranking.features[k]?.toFixed(2)}</td>
                  <td className="num">+{v.toFixed(3)}</td>
                </tr>
              ))}
            <tr>
              <td>Rank score ({item.ranking.ranker_version})</td>
              <td />
              <td className="num">{item.ranking.score.toFixed(3)}</td>
            </tr>
          </tbody>
        </table>
      ) : null}
      <div className="actions">
        {sent ? (
          <span className="ok small">Thanks for the feedback</span>
        ) : (
          <>
            <button type="button" onClick={() => send("recommendation_useful")} title="Useful recommendation">
              👍 Useful
            </button>
            <button type="button" onClick={() => send("recommendation_not_useful")} title="Not useful">
              👎
            </button>
            <button type="button" onClick={() => send("not_interested")} title="Don't show this again">
              Not interested
            </button>
          </>
        )}
        <button type="button" className="link" onClick={() => setWhy(!why)}>
          {why ? "Hide" : "Why?"}
        </button>
        {item.video.url ? (
          <a href={item.video.url} target="_blank" rel="noreferrer" className="link">
            Open ↗
          </a>
        ) : null}
      </div>
    </li>
  );
}

/** Items whose quality comes from a heuristic, or that carry review caveats, can be hidden. */
export function isReliable(item: RecommendationItem): boolean {
  const s = item.intrinsic_quality?.assessment_state;
  return s === "grounded";
}

export function DiscoverView({ onSelect }: { onSelect: Select }) {
  const [items, setItems] = useState<RecommendationItem[] | null>(null);
  const [notes, setNotes] = useState<string[]>([]);
  const [ineligible, setIneligible] = useState<Record<string, number>>({});
  const [onlyReliable, setOnlyReliable] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const load = useCallback(() => {
    setItems(null);
    setError(null);
    api
      .recommendations(20)
      .then((r) => {
        setItems(r.items);
        setNotes(r.notes);
        setIneligible(r.eligibility?.ineligible ?? {});
      })
      .catch(setError);
  }, []);
  useEffect(load, [load]);
  if (error) return <ErrorBox error={error} onRetry={load} />;
  if (!items) return <Spinner />;
  const shown = onlyReliable ? items.filter(isReliable) : items;
  const dropped = Object.values(ineligible).reduce((a, b) => a + b, 0);
  return (
    <div>
      <div className="toolbar">
        <span className="muted small">Ranked for you · quality is popularity-blind</span>
        <label className="check small">
          <input type="checkbox" checked={onlyReliable} onChange={(e) => setOnlyReliable(e.target.checked)} />
          Evidence-checked only
        </label>
        <button type="button" className="link" onClick={load}>
          Refresh
        </button>
      </div>
      {notes.map((n) => (
        <div key={n} className="muted small note">
          {n}
        </div>
      ))}
      {dropped > 0 ? (
        <div className="muted small" title={Object.entries(ineligible).map(([k, v]) => `${humanize(k)}: ${v}`).join("\n")}>
          {dropped} candidate{dropped === 1 ? "" : "s"} excluded by your filters or because their assessment was unusable.
        </div>
      ) : null}
      {shown.length === 0 ? (
        <Empty>
          {items.length > 0 && onlyReliable
            ? "None of the current recommendations has an evidence-checked assessment yet."
            : notes[0] ?? "No recommendations yet. Add interests in Preferences."}
        </Empty>
      ) : (
        <ul className="list">
          {shown.map((it) => (
            <RecCard key={it.video.id} item={it} onSelect={onSelect} />
          ))}
        </ul>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------------------- Video

function TranscriptForm({ videoId, onDone }: { videoId: string; onDone: () => void }) {
  const [content, setContent] = useState("");
  const [format, setFormat] = useState<"vtt" | "srt" | "text">("vtt");
  const [attestation, setAttestation] = useState("");
  const [state, setState] = useState<"idle" | "sending" | string | Error>("idle");
  return (
    <form
      className="report"
      onSubmit={(e) => {
        e.preventDefault();
        setState("sending");
        api
          .submitTranscript(videoId, { format, content, attestation })
          .then((r) => {
            setState(r.status);
            onDone();
          })
          .catch((err: Error) => setState(err));
      }}
    >
      <p className="small">
        Submit a transcript you are allowed to share (for example as the creator, or under a licence that
        permits this use). It is queued for evaluation by the backend.
      </p>
      <label>
        Format
        <select value={format} onChange={(e) => setFormat(e.target.value as "vtt" | "srt" | "text")}>
          <option value="vtt">WebVTT</option>
          <option value="srt">SRT</option>
          <option value="text">Plain text (timings estimated)</option>
        </select>
      </label>
      <label>
        Transcript
        <textarea rows={5} value={content} onChange={(e) => setContent(e.target.value)} required />
      </label>
      <label>
        Your right to submit it
        <input
          value={attestation}
          onChange={(e) => setAttestation(e.target.value)}
          placeholder="e.g. I am the creator of this video"
          minLength={10}
          required
        />
      </label>
      <button type="submit" disabled={state === "sending"}>
        Submit transcript
      </button>
      {state instanceof Error ? <ErrorBox error={state} /> : typeof state === "string" && state !== "idle" && state !== "sending" ? <div className="ok small">{state}</div> : null}
    </form>
  );
}

export function VideoView({ videoId }: { videoId: string | null }) {
  const [ev, setEv] = useState<Evaluation | null>(null);
  const [card, setCard] = useState<VideoCard | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [missing, setMissing] = useState<string | null>(null);
  useEffect(() => {
    if (!videoId) return;
    setEv(null);
    setCard(null);
    setError(null);
    setMissing(null);
    api
      .video(videoId)
      .then((v) => setCard(v.video))
      .catch(() => undefined);
    api
      .evaluation(videoId)
      .then(setEv)
      .catch((e: unknown) => (e instanceof ApiError && e.status === 404 ? setMissing(e.message) : setError(e)));
  }, [videoId]);
  if (!videoId) return <Empty>Open a YouTube video, or pick one from Discover or Search.</Empty>;
  if (error) return <ErrorBox error={error} />;
  return (
    <div>
      {card ? (
        <div className="video-head">
          <strong>{card.title}</strong>
          <VideoMeta v={card} />
        </div>
      ) : null}
      {missing ? (
        <>
          <AssessmentBadge state="not_evaluated" />
          <Empty>
            Not analyzed yet ({missing}). Evaluations require authorized content such as a creator-provided or
            licensed transcript.
          </Empty>
          <TranscriptForm videoId={videoId} onDone={() => undefined} />
        </>
      ) : ev ? (
        <QualityBreakdown ev={ev} videoUrl={card?.url ?? null} />
      ) : (
        <Spinner />
      )}
    </div>
  );
}

// --------------------------------------------------------------------------------------- Search

export function SearchView({ onSelect }: { onSelect: Select }) {
  const [q, setQ] = useState("");
  const [minQ, setMinQ] = useState(0);
  const [items, setItems] = useState<SearchItem[] | null>(null);
  const [error, setError] = useState<unknown>(null);
  return (
    <div>
      <form
        className="searchbar"
        onSubmit={(e) => {
          e.preventDefault();
          setError(null);
          api
            .search(q, { minQuality: minQ || undefined })
            .then((r) => setItems(r.items))
            .catch(setError);
        }}
      >
        <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Topic, e.g. hash tables" minLength={2} required />
        <select value={minQ} onChange={(e) => setMinQ(Number(e.target.value))} aria-label="Minimum quality">
          <option value={0}>Any quality</option>
          <option value={3}>≥ 3.0</option>
          <option value={3.5}>≥ 3.5</option>
          <option value={4}>≥ 4.0</option>
        </select>
        <button type="submit">Search</button>
      </form>
      {error ? <ErrorBox error={error} /> : null}
      {items === null ? (
        <Empty>Search the analyzed catalog by topic.</Empty>
      ) : items.length === 0 ? (
        <Empty>No analyzed videos match.</Empty>
      ) : (
        <ul className="list">
          {items.map((it) => (
            <li key={it.video.id} className="card">
              <div className="card-head">
                <button type="button" className="title" onClick={() => onSelect(it.video.id)}>
                  {it.video.title}
                </button>
                <QualityPill q={it.intrinsic_quality} />
              </div>
              <VideoMeta v={it.video} />
              <div className="muted small">match {it.match.score.toFixed(2)}</div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------------- Preferences

const PRIORITY_DIMS = ["depth", "factual_accuracy", "clarity", "information_density", "originality", "storytelling", "audio_quality"];
// v1 genres; each v2 genre (lecture, programming, ...) inherits its parent's weight unless set.
const GENRES = ["science", "education", "technology", "howto", "documentary", "news_commentary", "gaming", "entertainment", "art", "music"];
const V2_GENRES = ["lecture", "programming", "product_review", "interview_podcast", "comedy", "personal_storytelling"];
const PARENT: Record<string, string | undefined> = {
  lecture: "education",
  programming: "technology",
  product_review: "technology",
  comedy: "entertainment",
  personal_storytelling: "entertainment",
};
const INTENTS: { value: NonNullable<Preferences["intent"]>; label: string }[] = [
  { value: "any", label: "No preference" },
  { value: "learn", label: "Learn something" },
  { value: "stay_informed", label: "Stay informed" },
  { value: "be_entertained", label: "Be entertained" },
  { value: "deep_dive", label: "Deep dive" },
];

export function PreferencesView({ onSaved }: { onSaved: () => void }) {
  const [p, setP] = useState<Preferences | null>(null);
  const [interests, setInterests] = useState("");
  const [apiBase, setBase] = useState(DEFAULT_API_BASE);
  const [state, setState] = useState<"idle" | "saved" | Error>("idle");
  useEffect(() => {
    void getApiBase().then(setBase);
    api
      .preferences()
      .then((prefs) => {
        setP(prefs);
        setInterests(prefs.interests.join("\n"));
      })
      .catch((e: Error) => setState(e));
  }, []);
  const set = <K extends keyof Preferences>(k: K, v: Preferences[K]) => p && setP({ ...p, [k]: v });
  const saveBase = async () => {
    const url = apiBase.replace(/\/+$/, "");
    if (typeof chrome !== "undefined" && chrome.permissions && url !== DEFAULT_API_BASE) {
      const granted = await chrome.permissions.request({ origins: [`${new URL(url).origin}/*`] });
      if (!granted) return setState(new Error("Permission to contact that backend was not granted"));
    }
    await setApiBase(url);
    window.location.reload();
  };
  return (
    <div className="prefs">
      {state instanceof Error ? <ErrorBox error={state} /> : null}
      {p ? (
        <form
          onSubmit={(e) => {
            e.preventDefault();
            const body = { ...p, interests: interests.split("\n").map((s) => s.trim()).filter(Boolean) };
            api
              .savePreferences(body)
              .then((saved) => {
                setP(saved);
                setState("saved");
                onSaved();
              })
              .catch((err: Error) => setState(err));
          }}
        >
          <label>
            Interests (one per line)
            <textarea rows={4} value={interests} onChange={(e) => setInterests(e.target.value)} />
          </label>
          <label>
            Depth
            <select value={p.depth_preference} onChange={(e) => set("depth_preference", e.target.value as Preferences["depth_preference"])}>
              <option value="any">Any</option>
              <option value="intro">Introductory</option>
              <option value="intermediate">Intermediate</option>
              <option value="advanced">Advanced</option>
            </select>
          </label>
          <label>
            What are you here for? (used when no genre priorities are set)
            <select value={p.intent ?? "any"} onChange={(e) => set("intent", e.target.value as Preferences["intent"])}>
              {INTENTS.map((i) => (
                <option key={i.value} value={i.value}>
                  {i.label}
                </option>
              ))}
            </select>
          </label>
          <div className="row">
            <label>
              Min minutes
              <input type="number" min={1} max={600} value={p.min_duration_minutes ?? ""}
                onChange={(e) => set("min_duration_minutes", e.target.value ? Number(e.target.value) : null)} />
            </label>
            <label>
              Max minutes
              <input type="number" min={1} max={600} value={p.max_duration_minutes ?? ""}
                onChange={(e) => set("max_duration_minutes", e.target.value ? Number(e.target.value) : null)} />
            </label>
          </div>
          <label>
            Languages (comma-separated codes, empty = any)
            <input value={p.languages.join(", ")} onChange={(e) => set("languages", e.target.value.split(",").map((s) => s.trim()).filter(Boolean))} />
          </label>
          <fieldset>
            <legend>What matters to you (priority multipliers)</legend>
            {PRIORITY_DIMS.map((d) => (
              <label key={d} className="slider">
                <span>{humanize(d)}</span>
                <input type="range" min={0} max={3} step={0.5} value={p.dimension_weights[d] ?? 1}
                  onChange={(e) => set("dimension_weights", { ...p.dimension_weights, [d]: Number(e.target.value) })} />
                <span className="num">{fmtScore(p.dimension_weights[d] ?? 1)}×</span>
              </label>
            ))}
          </fieldset>
          <fieldset>
            <legend>Genres</legend>
            {[...GENRES, ...V2_GENRES].map((g) => {
              const inherited = p.genre_weights[g] === undefined && PARENT[g] !== undefined;
              const value = p.genre_weights[g] ?? (PARENT[g] ? p.genre_weights[PARENT[g]!] : undefined) ?? 0.5;
              return (
                <label key={g} className="slider">
                  <span>
                    {humanize(g)}
                    {inherited ? <span className="muted small"> (as {humanize(PARENT[g]!).toLowerCase()})</span> : null}
                  </span>
                  <input type="range" min={0} max={1} step={0.25} value={value}
                    onChange={(e) => set("genre_weights", { ...p.genre_weights, [g]: Number(e.target.value) })} />
                  <span className="num">{value.toFixed(2)}</span>
                </label>
              );
            })}
          </fieldset>
          <fieldset>
            <legend>Ranking</legend>
            <label className="slider">
              <span>Diversity</span>
              <input type="range" min={0} max={1} step={0.1} value={p.diversity} onChange={(e) => set("diversity", Number(e.target.value))} />
              <span className="num">{p.diversity.toFixed(1)}</span>
            </label>
            <label className="slider">
              <span>Minimum quality</span>
              <input type="range" min={0} max={4.5} step={0.5} value={p.min_quality} onChange={(e) => set("min_quality", Number(e.target.value))} />
              <span className="num">{p.min_quality.toFixed(1)}</span>
            </label>
            <label className="slider" title="Optional exposure boost for micro/small channels. It never changes quality scores.">
              <span>Small-creator boost</span>
              <input type="range" min={0} max={0.3} step={0.05} value={p.small_creator_boost} onChange={(e) => set("small_creator_boost", Number(e.target.value))} />
              <span className="num">{p.small_creator_boost.toFixed(2)}</span>
            </label>
            <label className="check">
              <input type="checkbox" checked={p.include_synthetic} onChange={(e) => set("include_synthetic", e.target.checked)} />
              Include synthetic demo videos
            </label>
          </fieldset>
          <button type="submit">Save preferences</button>
          <span role="status" aria-live="polite">{state === "saved" ? <span className="ok small"> Saved</span> : null}</span>
        </form>
      ) : state === "idle" ? (
        <Spinner />
      ) : null}
      <fieldset>
        <legend>Backend</legend>
        <label>
          API base URL
          <input value={apiBase} onChange={(e) => setBase(e.target.value)} />
        </label>
        <button type="button" onClick={() => void saveBase()}>
          Use this backend
        </button>
      </fieldset>
    </div>
  );
}
