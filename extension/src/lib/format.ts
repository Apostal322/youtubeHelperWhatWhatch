import type { DimensionStatus } from "./types";

export const STATUS_LABEL: Record<DimensionStatus, string> = {
  scored: "Scored",
  not_applicable: "Not applicable for this genre",
  insufficient_evidence: "Insufficient evidence",
  requires_media: "Needs authorized media",
  failed: "Evaluation failed",
};

export function fmtScore(score: number | null | undefined): string {
  return score === null || score === undefined ? "—" : score.toFixed(1);
}

export function pct(x: number | null | undefined): string {
  return x === null || x === undefined ? "—" : `${Math.round(x * 100)}%`;
}

export function humanize(key: string): string {
  return key.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
}

export function duration(seconds: number | null | undefined): string {
  if (!seconds) return "";
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return m >= 60 ? `${Math.floor(m / 60)}h ${m % 60}m` : `${m}:${String(s).padStart(2, "0")}`;
}

export function timestamp(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return "";
  const s = Math.floor(seconds);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

/** Score band used for colour coding (1-5 scale). */
export function band(score: number | null | undefined): "none" | "low" | "mid" | "high" {
  if (score === null || score === undefined) return "none";
  if (score >= 3.75) return "high";
  if (score >= 2.75) return "mid";
  return "low";
}

/** Badge text for the toolbar icon: one decimal, or a marker when not evaluated. */
export function badgeText(overall: number | null | undefined, evaluated: boolean): string {
  if (!evaluated) return "";
  return overall === null || overall === undefined ? "~" : overall.toFixed(1);
}
