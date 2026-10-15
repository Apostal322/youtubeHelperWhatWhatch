// Types mirroring the backend API (docs/api.md). Intrinsic quality, relevance and ranking are
// deliberately separate objects.

export type DimensionStatus =
  | "scored"
  | "not_applicable"
  | "insufficient_evidence"
  | "requires_media"
  | "failed";

export interface QualitySummary {
  evaluation_id: string;
  overall: number | null;
  overall_confidence: number | null;
  coverage: number;
  groups: Record<string, number | null>;
  status: string;
  synthetic: boolean;
  mode: string;
  genre: string;
  stage: string;
}

export interface VideoCard {
  id: string;
  title: string;
  channel_title: string | null;
  duration_seconds: number | null;
  language: string | null;
  genre: string | null;
  topics: string[];
  published_at: string | null;
  is_synthetic: boolean;
  url: string | null;
  status: string;
}

export interface RecommendationItem {
  impression_id: string | null;
  position: number;
  video: VideoCard;
  intrinsic_quality: QualitySummary | null;
  relevance: { similarity: number; normalized: number; matched_interest: string | null };
  ranking: {
    score: number;
    ranker_version: string;
    personal_quality: number | null;
    features: Record<string, number>;
    contributions: Record<string, number>;
    diversified: boolean;
  };
  explanations: string[];
}

export interface RecommendationResponse {
  request_id: string | null;
  ranker_version: string;
  items: RecommendationItem[];
  notes: string[];
}

export interface Evidence {
  segment_index: number | null;
  start_seconds: number | null;
  end_seconds: number | null;
  quote: string | null;
  note: string | null;
  verified: boolean;
}

export interface DimensionResult {
  dimension: string;
  group: string;
  status: DimensionStatus;
  score: number | null;
  confidence: number | null;
  justification: string;
  method: string;
  model: string | null;
  missing_inputs: string[];
  evidence: Evidence[];
}

export interface GroupScore {
  group: string;
  score: number | null;
  confidence: number | null;
  coverage: number;
}

export interface Evaluation {
  evaluation_id: string;
  video_id: string;
  stage: string;
  framework_version: string;
  mode: string;
  synthetic: boolean;
  disclaimer: string;
  models: string[];
  genre: string;
  genre_confidence: number;
  status: string;
  overall: {
    score: number | null;
    confidence: number | null;
    coverage: number;
    abstain_reason: string | null;
    method: string;
    notes: string[];
  };
  groups: Record<string, GroupScore>;
  contributions: { dimension: string; effective_weight: number; score: number; contribution: number }[];
  dimensions: DimensionResult[];
  validation: { passed: boolean; warnings: string[] };
  missing_inputs: string[];
  created_at: string | null;
}

export interface SearchItem {
  video: VideoCard;
  intrinsic_quality: QualitySummary | null;
  match: { semantic_similarity: number; title_match: number; score: number };
}

export interface Preferences {
  interests: string[];
  genre_weights: Record<string, number>;
  dimension_weights: Record<string, number>;
  depth_preference: "any" | "intro" | "intermediate" | "advanced";
  languages: string[];
  max_duration_minutes: number | null;
  diversity: number;
  min_quality: number;
  small_creator_boost: number;
  include_synthetic: boolean;
}

export interface Meta {
  framework_version: string;
  demo_mode: boolean;
  evaluation_mode: string;
  disclaimer: string;
  llm_provider: string;
  dimensions: { dimension: string; group: string; question: string; anchors: Record<string, string> }[];
  genre_profiles: Record<string, { description: string }>;
}

export type FeedbackKind =
  | "recommendation_useful"
  | "recommendation_not_useful"
  | "not_interested"
  | "evaluation_incorrect";

export interface FeedbackBody {
  video_id: string;
  kind: FeedbackKind;
  impression_id?: string | null;
  evaluation_id?: string | null;
  dimension?: string | null;
  suggested_score?: number | null;
  comment?: string | null;
}
