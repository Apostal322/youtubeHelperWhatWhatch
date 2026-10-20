import type { ReactNode } from "react";
import { ApiError, friendlyMessage } from "../../lib/api";
import { ASSESSMENT, SOURCE, band, fmtScore, modelLabel } from "../../lib/format";
import type { AssessmentState, EvaluatorSource, QualitySummary } from "../../lib/types";

/** The kind of assessment, always spelled out in text (colour is only a secondary cue). */
// Reasons are listed only where they add information; for the other states the label says it all.
const CAVEAT_STATES = new Set<string>(["preliminary", "insufficient_evidence", "failed"]);

export function AssessmentBadge({ state, reasons }: { state: AssessmentState | "not_evaluated"; reasons?: string[] }) {
  const info = ASSESSMENT[state] ?? ASSESSMENT.preliminary;
  return (
    <div className={`assessment assessment-${info.tone}`} role="note" aria-label={`Assessment: ${info.label}`}>
      <strong>{info.label}</strong> <span className="small">{info.explain}</span>
      {reasons && reasons.length > 0 && CAVEAT_STATES.has(state) ? (
        <ul className="small reasons">
          {reasons.slice(0, 4).map((r) => (
            <li key={r}>{r}</li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

/** Who produced the scores (local model, hosted model, heuristic, synthetic), with the model and
 * the number of independent human ratings when there are any. */
export function SourceLine({ source, models, humanReviews }: { source?: EvaluatorSource; models?: string[]; humanReviews?: number }) {
  if (!source) return null;
  const info = SOURCE[source];
  const shown = source === "local_ai" || source === "hosted_ai" ? (models ?? []).filter((m) => m !== "heuristic-baseline").map(modelLabel) : [];
  return (
    <div className={`source source-${source}`} role="note" aria-label={`Evaluated by: ${info.label}`}>
      <strong>{info.label}</strong>
      {shown.length ? <span className="small"> · {shown.join(", ")}</span> : null} <span className="small muted">{info.explain}</span>
      {humanReviews ? (
        <div className="small">
          <strong>Human-reviewed:</strong> {humanReviews} independent rating{humanReviews === 1 ? "" : "s"} (kept separate from the AI scores).
        </div>
      ) : null}
    </div>
  );
}

export function ScoreBar({ score, label, sub }: { score: number | null; label: string; sub?: string }) {
  const width = score === null ? 0 : ((score - 1) / 4) * 100;
  return (
    <div className="scorebar">
      <div className="scorebar-head">
        <span>{label}</span>
        <span className={`score score-${band(score)}`}>{fmtScore(score)}</span>
      </div>
      <div className="track" role="meter" aria-valuemin={1} aria-valuemax={5} aria-valuenow={score ?? undefined} aria-label={label}>
        <div className={`fill fill-${band(score)}`} style={{ width: `${width}%` }} />
      </div>
      {sub ? <div className="muted small">{sub}</div> : null}
    </div>
  );
}

export function QualityPill({ q }: { q: QualitySummary | null }) {
  if (!q) return <span className="pill pill-none">not evaluated</span>;
  const state = q.assessment_state;
  const caveat = state && state !== "grounded" ? ASSESSMENT[state]?.label : null;
  const source = q.evaluator_source ? SOURCE[q.evaluator_source].label : null;
  if (q.overall === null) {
    return (
      <span className="pill pill-none" title="Not enough evidence for a genre-weighted overall score; see the breakdown">
        partial
      </span>
    );
  }
  return (
    <span
      className={`pill pill-${band(q.overall)}`}
      title={`Overall (genre-weighted), confidence ${Math.round((q.overall_confidence ?? 0) * 100)}%${caveat ? ` · ${caveat}` : ""}${source ? ` · ${source}` : ""}`}
    >
      {q.overall.toFixed(1)}
      {caveat ? <span className="pill-caveat"> · {caveat.toLowerCase()}</span> : null}
      {q.evaluator_source === "local_ai" ? <span className="pill-caveat"> · local AI</span> : null}
    </span>
  );
}

export function ModeBanner({ synthetic, text }: { synthetic: boolean; text: string }) {
  return <div className={synthetic ? "banner banner-warn" : "banner"}>{text}</div>;
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>;
}

export function ErrorBox({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  const requestId = error instanceof ApiError ? error.requestId : null;
  return (
    <div className="error" role="alert">
      <span aria-hidden="true">⚠ </span>
      {friendlyMessage(error)}
      {requestId ? <div className="small muted">Request ID: {requestId}</div> : null}
      {onRetry ? (
        <button type="button" className="link" onClick={onRetry}>
          Try again
        </button>
      ) : null}
    </div>
  );
}

export function Spinner() {
  return (
    <div className="muted small" role="status" aria-live="polite">
      Loading…
    </div>
  );
}
