import { useState } from "react";
import { api } from "../../lib/api";
import { STATUS_LABEL, fmtScore, humanize, pct, timestamp } from "../../lib/format";
import type { DimensionResult, Evaluation } from "../../lib/types";
import { ErrorBox, ModeBanner, ScoreBar } from "./common";

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
                <li key={i}>
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
      <ModeBanner synthetic={ev.synthetic} text={ev.disclaimer} />
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
      <ReportIncorrect ev={ev} />
    </div>
  );
}
