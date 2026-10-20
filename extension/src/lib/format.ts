import type { AssessmentState, ClaimVerdict, DimensionStatus, Evaluation, EvaluatorSource } from "./types";

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

/** A link to the moment in the video, only where the platform supports it (YouTube's `t`
 * parameter). Other pages, such as a podcast's episode page, get no deep link. */
export function timeLink(videoUrl: string | null, seconds: number): string | null {
  if (!videoUrl) return null;
  try {
    const u = new URL(videoUrl);
    if (!/(^|\.)youtube\.com$/.test(u.hostname) && u.hostname !== "youtu.be") return null;
    u.searchParams.set("t", `${Math.floor(seconds)}s`);
    return u.toString();
  } catch {
    return null;
  }
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

/** User-facing meaning of each assessment state. ``tone`` drives styling; the label always carries
 * the meaning in text (never colour alone). */
export const ASSESSMENT: Record<AssessmentState | "not_evaluated", { label: string; tone: "ok" | "warn" | "muted" | "bad"; explain: string }> = {
  grounded: {
    label: "Evidence-checked",
    tone: "ok",
    explain: "AI judgements whose cited quotes were verified against the transcript. Not yet validated against human raters.",
  },
  preliminary: {
    label: "Preliminary",
    tone: "warn",
    explain: "Scores exist but with caveats (review flags, unverified or missing evidence, partial coverage or low confidence).",
  },
  insufficient_evidence: {
    label: "Insufficient evidence",
    tone: "muted",
    explain: "Too little could be judged from the authorized material to give scores.",
  },
  failed: { label: "Evaluation failed", tone: "bad", explain: "The evaluators failed; no scores are shown." },
  heuristic_baseline: {
    label: "Heuristic baseline",
    tone: "warn",
    explain: "Keyword and structure heuristics on real content. Not an AI or human assessment.",
  },
  synthetic_demo: {
    label: "Synthetic demo",
    tone: "warn",
    explain: "A heuristic run over synthetic demo content. Not a real video and not an AI assessment.",
  },
  not_evaluated: {
    label: "Not evaluated",
    tone: "muted",
    explain: "No authorized content has been evaluated for this video yet.",
  },
};

/** Who produced the scores. Shown next to the assessment state, always as text. */
export const SOURCE: Record<EvaluatorSource, { label: string; explain: string }> = {
  local_ai: {
    label: "Local AI evaluation",
    explain: "Judged by an open-weight model running on this service's own hardware; transcripts are not sent to a cloud API.",
  },
  hosted_ai: { label: "Hosted AI evaluation", explain: "Judged by a model provider's API." },
  heuristic: { label: "Heuristic", explain: "Keyword and structure rules, not a model." },
  synthetic: { label: "Synthetic", explain: "Demo content and a heuristic run; not a real assessment." },
};

/** The model as shown to people: ``ollama:qwen3.5:4b@ab12…:Q4_K_M`` → ``qwen3.5:4b (Q4_K_M)``. */
export function modelLabel(served: string): string {
  const local = served.startsWith("ollama:") ? served.slice("ollama:".length) : served;
  const m = /^(.+?)@[^:]*:(.+)$/.exec(local);
  return m ? `${m[1]} (${m[2]})` : local;
}

export const VERDICT_LABEL: Record<ClaimVerdict, string> = {
  supported: "Supported",
  contradicted: "Contradicted",
  mixed: "Partly contradicted",
  outdated: "Accurate when published, since changed",
  unverifiable: "Not in reference sources",
  insufficient_evidence: "Undecided",
  not_applicable: "Not a factual claim",
  unchecked: "Not checked",
};

/** A plain-language, factual summary of an evaluation: strongest and weakest scored dimensions,
 * and what could not be judged (and why). Built only from fields in the evaluation. */
export function plainSummary(ev: Evaluation): string {
  const scored = ev.dimensions.filter((d) => d.status === "scored" && d.score !== null && d.group !== "technical");
  if (scored.length === 0) return "Nothing about the content could be judged from the available material.";
  const sorted = [...scored].sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
  const top = sorted.filter((d) => (d.score ?? 0) >= 3.5).slice(0, 2);
  const low = sorted.filter((d) => (d.score ?? 0) <= 2.5).slice(-2);
  const parts: string[] = [];
  if (top.length) parts.push(`Strongest: ${top.map((d) => humanize(d.dimension).toLowerCase()).join(" and ")}.`);
  if (low.length) parts.push(`Weakest: ${low.map((d) => humanize(d.dimension).toLowerCase()).join(" and ")}.`);
  if (!top.length && !low.length) parts.push("Mid-range on every scored aspect.");
  const media = ev.dimensions.filter((d) => d.status === "requires_media").map((d) => humanize(d.dimension).toLowerCase());
  const unknown = ev.dimensions.filter((d) => d.status === "insufficient_evidence").map((d) => humanize(d.dimension).toLowerCase());
  if (unknown.length) parts.push(`Not judged (too little evidence): ${unknown.join(", ")}.`);
  if (media.length) parts.push(`Not judged (needs authorized audio/video): ${media.join(", ")}.`);
  return parts.join(" ");
}
