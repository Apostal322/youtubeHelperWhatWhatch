import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { QualityBreakdown } from "../src/popup/components/QualityBreakdown";
import type { Evaluation } from "../src/lib/types";

const ev: Evaluation = {
  evaluation_id: "e1",
  video_id: "syn_sci_001",
  stage: "B",
  framework_version: "2026.10.0",
  mode: "synthetic_demo",
  synthetic: true,
  disclaimer: "Synthetic demo evaluation ... NOT an AI assessment",
  models: ["heuristic-baseline"],
  genre: "science",
  genre_confidence: 0.8,
  status: "complete",
  overall: { score: 3.4, confidence: 0.4, coverage: 0.7, abstain_reason: null, method: "m", notes: [] },
  groups: {
    content: { group: "content", score: 3.2, confidence: 0.4, coverage: 0.6 },
    technical: { group: "technical", score: null, confidence: null, coverage: 0 },
  },
  contributions: [],
  dimensions: [
    {
      dimension: "depth", group: "content", status: "scored", score: 4, confidence: 0.4,
      justification: "Explains mechanisms.", method: "heuristic_baseline_v1", model: null, missing_inputs: [],
      evidence: [{ segment_index: 3, start_seconds: 65, end_seconds: 70, quote: "because hydrogen bonds", note: null, verified: true }],
    },
    {
      dimension: "originality", group: "content", status: "insufficient_evidence", score: null, confidence: null,
      justification: "Cannot judge.", method: "heuristic_baseline_v1", model: null, missing_inputs: [], evidence: [],
    },
    {
      dimension: "audio_quality", group: "technical", status: "requires_media", score: null, confidence: null,
      justification: "No audio.", method: "pipeline", model: null, missing_inputs: ["audio"], evidence: [],
    },
  ],
  validation: { passed: true, warnings: [] },
  missing_inputs: ["audio"],
  created_at: null,
};

describe("QualityBreakdown", () => {
  it("labels synthetic evaluations and shows abstentions without scores", () => {
    render(<QualityBreakdown ev={ev} videoUrl={null} />);
    expect(screen.getByText(/NOT an AI assessment/)).toBeTruthy();
    expect(screen.getByText("Insufficient evidence")).toBeTruthy();
    expect(screen.getByText("Needs authorized media")).toBeTruthy();
    fireEvent.click(screen.getByText("Depth"));
    expect(screen.getByText("because hydrogen bonds")).toBeTruthy();
    expect(screen.getByText("1:05")).toBeTruthy();
  });

  it("explains an abstained overall score", () => {
    render(<QualityBreakdown ev={{ ...ev, overall: { ...ev.overall, score: null, abstain_reason: "content coverage too low" } }} videoUrl={null} />);
    expect(screen.getByText(/No overall score: content coverage too low/)).toBeTruthy();
  });

  it("labels a local-model evaluation with the model and keeps human ratings separate", () => {
    const local: Evaluation = {
      ...ev,
      mode: "llm",
      synthetic: false,
      disclaimer: "AI evaluation",
      models: [`ollama:qwen3.5:4b@${"ab".repeat(6)}:Q4_K_M`],
      assessment_state: "grounded",
      evaluator_source: "local_ai",
      human_reviews: 2,
    };
    render(<QualityBreakdown ev={local} videoUrl={null} />);
    expect(screen.getByText("Local AI evaluation")).toBeTruthy();
    expect(screen.getByText(/qwen3\.5:4b \(Q4_K_M\)/)).toBeTruthy();
    expect(screen.getByText(/2 independent ratings/)).toBeTruthy();
    expect(screen.getByText("Evidence-checked")).toBeTruthy();
  });

  it("does not call a heuristic run AI", () => {
    render(<QualityBreakdown ev={{ ...ev, evaluator_source: "synthetic" }} videoUrl={null} />);
    expect(screen.getByText("Synthetic")).toBeTruthy();
    expect(screen.queryByText("Local AI evaluation")).toBeNull();
  });
});
