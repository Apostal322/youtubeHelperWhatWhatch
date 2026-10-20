import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiClient, ApiError, REQUEST_TIMEOUT_MS, api, friendlyMessage } from "../src/lib/api";
import { ASSESSMENT, plainSummary } from "../src/lib/format";
import type { Evaluation, RecommendationItem } from "../src/lib/types";
import { ClaimsPanel, QualityBreakdown } from "../src/popup/components/QualityBreakdown";
import { genreText, isReliable } from "../src/popup/components/Views";
import { AssessmentBadge } from "../src/popup/components/common";

const dim = (dimension: string, group: string, status: Evaluation["dimensions"][number]["status"], score: number | null) => ({
  dimension, group, status, score, confidence: score === null ? null : 0.6, justification: "j", method: "llm_rubric",
  model: "m", missing_inputs: [], evidence: [],
});

const ev: Evaluation = {
  evaluation_id: "e1", video_id: "v1", stage: "B", framework_version: "2026.10.1", mode: "llm", synthetic: false,
  disclaimer: "Automated LLM-based assessment.", models: ["m"], genre: "lecture", genre_confidence: 0.7, status: "complete",
  overall: { score: 3.6, confidence: 0.6, coverage: 0.8, abstain_reason: null, method: "m", notes: [] },
  groups: { content: { group: "content", score: 3.8, confidence: 0.6, coverage: 0.9 } },
  contributions: [],
  dimensions: [
    { ...dim("depth", "content", "scored", 4.5), evidence: [
      { kind: "supporting", segment_index: 1, start_seconds: 10, end_seconds: 20, quote: "the enzyme fixes carbon", note: null, verified: true },
      { kind: "counter", segment_index: 2, start_seconds: 30, end_seconds: 40, quote: "skips the light reactions", note: null, verified: true },
    ] },
    dim("clarity", "presentation", "scored", 4.0),
    dim("pacing", "presentation", "scored", 2.0),
    dim("originality", "content", "insufficient_evidence", null),
    dim("visual_quality", "technical", "requires_media", null),
    dim("audio_quality", "technical", "scored", 1.5),
  ],
  validation: { passed: true, warnings: [] },
  missing_inputs: [],
  created_at: null,
  assessment_state: "preliminary",
  assessment_state_reasons: ["model samples disagree on: pacing"],
};

afterEach(() => vi.useRealTimers());

describe("plainSummary", () => {
  it("states strengths, weaknesses and what was not judged, ignoring technical scores", () => {
    const s = plainSummary(ev);
    expect(s).toContain("Strongest: depth and clarity.");
    expect(s).toContain("Weakest: pacing.");
    expect(s).toContain("Not judged (too little evidence): originality.");
    expect(s).toContain("Not judged (needs authorized audio/video): visual quality.");
    expect(s).not.toContain("audio quality"); // production quality is not part of the content summary
  });
  it("says so when nothing could be judged", () => {
    expect(plainSummary({ ...ev, dimensions: [dim("depth", "content", "insufficient_evidence", null)] })).toMatch(/Nothing about the content/);
  });
});

describe("assessment states", () => {
  it("spells out every state in text", () => {
    for (const [state, info] of Object.entries(ASSESSMENT)) {
      const { unmount } = render(<AssessmentBadge state={state as keyof typeof ASSESSMENT} reasons={["r1"]} />);
      expect(screen.getByRole("note", { name: `Assessment: ${info.label}` })).toBeTruthy();
      unmount();
    }
  });
  it("shows caveats for preliminary results but not for evidence-checked ones", () => {
    const { unmount } = render(<AssessmentBadge state="preliminary" reasons={["evidence grounding failed"]} />);
    expect(screen.getByText("evidence grounding failed")).toBeTruthy();
    unmount();
    render(<AssessmentBadge state="grounded" reasons={["n/a"]} />);
    expect(screen.queryByText("n/a")).toBeNull();
  });
  it("only evidence-checked items pass the reliability filter", () => {
    const item = (s?: string) => ({ intrinsic_quality: s ? { assessment_state: s } : null }) as unknown as RecommendationItem;
    expect([item("grounded"), item("preliminary"), item("heuristic_baseline"), item()].map(isReliable)).toEqual([true, false, false, false]);
  });
});

describe("genre status", () => {
  it("qualifies uncertain and corrected genres only", () => {
    expect(genreText({ genre: "interview_podcast", genre_status: "confident" })).toBe("Interview podcast");
    expect(genreText({ genre: "lecture", genre_status: "overridden" })).toBe("Lecture (corrected)");
    expect(genreText({ genre: "other", genre_status: "ambiguous" })).toBe("Other (ambiguous)");
    expect(genreText({ genre: null, genre_status: "unknown" })).toBeNull();
  });
});

describe("QualityBreakdown v2", () => {
  it("shows the assessment, the summary and marks counter-evidence", () => {
    render(<QualityBreakdown ev={ev} videoUrl={null} />);
    expect(screen.getByRole("note", { name: "Assessment: Preliminary" })).toBeTruthy();
    expect(screen.getByText(/Strongest: depth and clarity/)).toBeTruthy();
    fireEvent.click(screen.getByText("Depth"));
    expect(screen.getByText("counter-evidence")).toBeTruthy();
    expect(screen.getByText("skips the light reactions")).toBeTruthy();
  });
});

describe("ClaimsPanel", () => {
  it("renders untrusted claim text as text and shows verdict provenance", async () => {
    const spy = vi.spyOn(api, "claims").mockResolvedValue({
      video_id: "v1",
      items: [
        { segment_index: 3, quote: "<img src=x onerror=alert(1)>", claim: "<b>bold claim</b>", claim_type: "factual_assertion",
          verdict: "contradicted", domain_risk: "high_stakes",
          sources: [{ passage_id: "p#0", doc_id: "d", title: "Reference", license: "CC-BY-4.0" }],
          explanation: "The reference says otherwise.", checked_at: null, stage: "C" },
      ],
    });
    const { container } = render(<ClaimsPanel videoId="v1" />);
    const details = container.querySelector("details")!;
    details.open = true;
    fireEvent(details, new Event("toggle"));
    await waitFor(() => expect(screen.getByText("Contradicted")).toBeTruthy());
    expect(screen.getByText("<b>bold claim</b>")).toBeTruthy();
    expect(container.querySelector("img")).toBeNull();
    expect(container.querySelector("b")).toBeNull();
    expect(screen.getByText("high stakes")).toBeTruthy();
    expect(screen.getByText(/Reference \(CC-BY-4.0\)/)).toBeTruthy();
    spy.mockRestore();
  });
});

describe("API errors", () => {
  const json = (status: number, body: unknown, headers: Record<string, string> = {}) =>
    new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json", ...headers } });

  it("parses the structured error shape, request id and Retry-After", async () => {
    const client = new ApiClient(async () =>
      json(429, { detail: "rate limit exceeded; retry later", error: { code: "rate_limited", message: "rate limit exceeded; retry later", request_id: "req-abc12345" } }, { "Retry-After": "7" }),
    );
    const err = (await client.meta().catch((e: unknown) => e)) as ApiError;
    expect(err).toBeInstanceOf(ApiError);
    expect([err.status, err.code, err.requestId, err.retryAfterSeconds]).toEqual([429, "rate_limited", "req-abc12345", 7]);
    expect(friendlyMessage(err)).toBe("Too many requests; try again in 7s.");
  });

  it("explains expired sessions and blinded items, and drops a rejected token", async () => {
    const store: Record<string, unknown> = { "avd.token": "old" };
    vi.stubGlobal("chrome", {
      storage: {
        local: {
          get: async (k: string) => ({ [k]: store[k] }),
          set: async (o: Record<string, unknown>) => Object.assign(store, o),
          remove: async (k: string) => {
            delete store[k];
          },
        },
      },
    });
    const expired = new ApiClient(async () => json(401, { detail: "invalid token", error: { code: "unauthenticated", message: "invalid token", request_id: "r1" } }));
    const e1 = (await expired.recommendations().catch((e: unknown) => e)) as ApiError;
    expect(friendlyMessage(e1)).toMatch(/session expired/);
    expect(store["avd.token"]).toBeUndefined();
    const blinded = new ApiClient(async () =>
      json(403, { detail: "assessment withheld while blind human annotation of this item is open", error: { code: "forbidden", message: "assessment withheld while blind human annotation of this item is open", request_id: "r2" } }),
    );
    const e2 = (await blinded.evaluation("v1").catch((e: unknown) => e)) as ApiError;
    expect(friendlyMessage(e2)).toMatch(/withheld while independent human raters/);
    vi.unstubAllGlobals();
  });

  it("maps network failures and timeouts to distinct codes", async () => {
    const down = new ApiClient(async () => {
      throw new TypeError("Failed to fetch");
    });
    const e1 = (await down.meta().catch((e: unknown) => e)) as ApiError;
    expect(e1.code).toBe("network");
    expect(friendlyMessage(e1)).toMatch(/Cannot reach the backend/);

    vi.useFakeTimers();
    const hang = new ApiClient(
      (_url, init) =>
        new Promise((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
        }),
    );
    const pending = hang.meta().catch((e: unknown) => e);
    await vi.advanceTimersByTimeAsync(REQUEST_TIMEOUT_MS + 1);
    const e2 = (await pending) as ApiError;
    expect(e2.code).toBe("timeout");
  });
});
