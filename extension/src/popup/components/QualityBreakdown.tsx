import { useState } from "react";
import { api } from "../../lib/api";
import { STATUS_LABEL, VERDICT_LABEL, fmtScore, humanize, pct, plainSummary, timestamp } from "../../lib/format";
import type { Claim, DimensionResult, Evaluation } from "../../lib/types";
import { AssessmentBadge, ErrorBox, ModeBanner, ScoreBar, Spinner } from "./common";

/** Claims extracted from the transcript and, where checked, their verdicts from licensed reference
 * passages (never from model knowledge). Loaded on demand. */
export function ClaimsPanel({ videoId }: { videoId: string }) {
  const [claims, setClaims] = useState<Claim[] | null>(null);
  const [error, setError] = useState<unknown>(null);
  const load = () => {
    setError(null);
    api
      .claims(videoId)
      .then((r) => setClaims(r.items))
      .catch(setError);
  };
  return (
    <details
      className="claims"
      onToggle={(e) => {
        if ((e.target as HTMLDetailsElement).open && claims === null) load();
      }}
    >
      <summary>Claims and fact checks</summary>
      {error ? <ErrorBox error={error} onRetry={load} /> : null}
      {claims === null && !error ? <Spinner /> : null}
      {claims !== null && claims.length === 0 ? <div className="muted small">No claims were extracted.</div> : null}
      {claims !== null && claims.length > 0 ? (
        <ul className="claim-list">
          {claims.map((c, i) => (
            <li key={`${c.segment_index}-${i}`} className={`claim claim-${c.verdict}`}>
              <span className="verdict">{VERDICT_LABEL[c.verdict] ?? c.verdict}</span>
              {c.domain_risk === "high_stakes" ? <span className="tag">high stakes</span> : null}{" "}
              <span>{c.claim}</span>
              <div className="muted small">
                {humanize(c.claim_type)}
                {c.quote ? (
                  <>
                    {" · "}
                    <q>{c.quote}</q>
                  </>
                ) : null}
              </div>
              {c.sources.length > 0 ? (
                <div className="muted small">
                  Sources: {c.sources.map((s) => `${s.title} (${s.license})`).join("; ")}
                </div>
              ) : null}
              {c.explanation ? <div className="small">{c.explanation}</div> : null}
            </li>
          ))}
        </ul>
      ) : null}
    </details>
  );
}

const GROUP_ORDER = ["content", "presentation", "technical"];

function DimensionRow({ d, videoUrl }: { d: DimensionResult; videoUrl: string | null }) {
  const [open, setOpen] = useState(false);
  const scored = d.status === "scored";
  return (
    <li className={`dim ${scored ? "" : "dim-abstain"}`}>
      <button type="button" className="dim-head" onClick={() => setOpen(!open)} aria-expanded={open}>
        <span>{humanize(d.dimension)}</span>
        {scored ? (
          <span className="dim-score">
            {fmtScore(d.score)} <span className="muted small">conf {pct(d.confidence)}</span>
          </span>
        ) : (
          <span className="muted small">{STATUS_LABEL[d.status]}</span>
        )}
      </button>
      {open ? (
        <div className="dim-body">
          <p>{d.justification}</p>
          {d.evidence.length > 0 ? (
            <ul className="evidence">
              {d.evidence.map((e, i) => (
                <li key={i} className={e.kind === "counter" ? "evidence-counter" : undefined}>
                  {e.kind === "counter" ? <span className="tag">counter-evidence</span> : null}
                  {e.start_seconds !== null ? (
                    videoUrl ? (
                      <a href={`${videoUrl}&t=${Math.floor(e.start_seconds)}s`} target="_blank" rel="noreferrer">
                        {timestamp(e.start_seconds)}
                      </a>
                    ) : (
                      <span className="ts">{timestamp(e.start_seconds)}</span>
                    )
                  ) : null}{" "}
                  {e.quote ? <q>{e.quote}</q> : null} {e.note ? <span className="muted small">{e.note}</span> : null}
                </li>
              ))}
            </ul>
          ) : null}
          {d.missing_inputs.length > 0 ? (
            <div className="muted small">Missing inputs: {d.missing_inputs.join(", ")}</div>
          ) : null}
          <div className="muted small">
            Method: {d.method}
            {d.model ? ` · ${d.model}` : ""}
          </div>
        </div>
      ) : null}
    </li>
  );
}

function ReportIncorrect({ ev }: { ev: Evaluation }) {
  const [open, setOpen] = useState(false);
  const [dimension, setDimension] = useState("");
  const [score, setScore] = useState("");
  const [comment, setComment] = useState("");
  const [state, setState] = useState<"idle" | "sending" | "sent" | Error>("idle");
  if (!open) {
    return (
      <button type="button" className="link" onClick={() => setOpen(true)}>
        Report an incorrect evaluation
      </button>
    );
  }
  if (state === "sent") return <div className="ok">Thanks — your report was recorded for review.</div>;
  return (
    <form
      className="report"
      onSubmit={(e) => {
        e.preventDefault();
        setState("sending");
        api
          .feedback({
            video_id: ev.video_id,
            kind: "evaluation_incorrect",
            evaluation_id: ev.evaluation_id,
            dimension: dimension || null,
            suggested_score: score ? Number(score) : null,
            comment: comment || null,
          })
          .then(() => setState("sent"))
          .catch((err: Error) => setState(err));
      }}
    >
      <label>
        Dimension
        <select value={dimension} onChange={(e) => setDimension(e.target.value)}>
          <option value="">Whole evaluation</option>
          {ev.dimensions.map((d) => (
            <option key={d.dimension} value={d.dimension}>
              {humanize(d.dimension)}
            </option>
          ))}
        </select>
      </label>
      <label>
        Score you'd expect (1–5, optional)
        <input type="number" min={1} max={5} step={0.5} value={score} onChange={(e) => setScore(e.target.value)} />
      </label>
      <label>
        What is wrong?
        <textarea value={comment} onChange={(e) => setComment(e.target.value)} rows={3} maxLength={2000} />
      </label>
      <button type="submit" disabled={state === "sending"}>
        Send report
      </button>
      {state instanceof Error ? <ErrorBox error={state} /> : null}
    </form>
  );
}

export function QualityBreakdown({ ev, videoUrl }: { ev: Evaluation; videoUrl: string | null }) {
  const groups = GROUP_ORDER.filter((g) => ev.groups[g]);
  return (
    <div className="breakdown">
      <AssessmentBadge state={ev.assessment_state ?? (ev.synthetic ? "synthetic_demo" : "preliminary")} reasons={ev.assessment_state_reasons} />
      <ModeBanner synthetic={ev.synthetic} text={ev.disclaimer} />
      <p className="summary">{plainSummary(ev)}</p>
      <div className="overall">
        <ScoreBar
          score={ev.overall.score}
          label={`Overall · ${humanize(ev.genre)} profile`}
          sub={
            ev.overall.score === null
              ? `No overall score: ${ev.overall.abstain_reason ?? "insufficient evidence"}`
              : `confidence ${pct(ev.overall.confidence)} · evidence coverage ${pct(ev.overall.coverage)}`
          }
        />
        {ev.overall.notes.map((n) => (
          <div key={n} className="muted small note">
            {n}
          </div>
        ))}
      </div>
      {groups.map((g) => {
        const gs = ev.groups[g]!;
        return (
          <section key={g} className="group">
            <ScoreBar score={gs.score} label={humanize(g)} sub={`coverage ${pct(gs.coverage)}`} />
            <ul className="dims">
              {ev.dimensions
                .filter((d) => d.group === g && d.status !== "not_applicable")
                .map((d) => (
                  <DimensionRow key={d.dimension} d={d} videoUrl={videoUrl} />
                ))}
            </ul>
          </section>
        );
      })}
      {ev.validation.warnings.length > 0 ? (
        <details className="muted small">
          <summary>Validation notes ({ev.validation.warnings.length})</summary>
          <ul>
            {ev.validation.warnings.map((w) => (
              <li key={w}>{w}</li>
            ))}
          </ul>
        </details>
      ) : null}
      <div className="muted small">
        Framework {ev.framework_version} · stage {ev.stage}
        {ev.models.length ? ` · ${ev.models.join(", ")}` : ""}
      </div>
      <ClaimsPanel videoId={ev.video_id} />
      <ReportIncorrect ev={ev} />
    </div>
  );
}
