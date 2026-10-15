import type { ReactNode } from "react";
import { band, fmtScore } from "../../lib/format";
import type { QualitySummary } from "../../lib/types";

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
  if (q.overall === null) {
    return (
      <span className="pill pill-none" title="Not enough evidence for a genre-weighted overall score; see the breakdown">
        partial
      </span>
    );
  }
  return (
    <span className={`pill pill-${band(q.overall)}`} title={`Overall (genre-weighted), confidence ${Math.round((q.overall_confidence ?? 0) * 100)}%`}>
      {q.overall.toFixed(1)}
    </span>
  );
}

export function ModeBanner({ synthetic, text }: { synthetic: boolean; text: string }) {
  return <div className={synthetic ? "banner banner-warn" : "banner"}>{text}</div>;
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>;
}

export function ErrorBox({ error }: { error: unknown }) {
  const msg = error instanceof Error ? error.message : String(error);
  return <div className="error">⚠ {msg}</div>;
}

export function Spinner() {
  return <div className="muted small">Loading…</div>;
}
