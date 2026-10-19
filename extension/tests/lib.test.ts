import { beforeEach, describe, expect, it, vi } from "vitest";
import { ApiClient, ApiError, setApiBase } from "../src/lib/api";
import { badgeText, band, duration, fmtScore, modelLabel, timestamp } from "../src/lib/format";
import { parseVideoId } from "../src/lib/youtube";

describe("parseVideoId", () => {
  it.each([
    ["https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=10s", "dQw4w9WgXcQ"],
    ["https://m.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"],
    ["https://youtu.be/dQw4w9WgXcQ?si=x", "dQw4w9WgXcQ"],
    ["https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"],
    ["https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"],
  ])("%s", (url, id) => expect(parseVideoId(url)).toBe(id));

  it.each([
    "https://www.youtube.com/",
    "https://www.youtube.com/watch?v=short",
    "https://evil.example/watch?v=dQw4w9WgXcQ",
    "not a url",
    undefined,
  ])("rejects %s", (url) => expect(parseVideoId(url)).toBeNull());
});

describe("format", () => {
  it("formats scores and bands", () => {
    expect(fmtScore(null)).toBe("—");
    expect(fmtScore(3.456)).toBe("3.5");
    expect(band(4.2)).toBe("high");
    expect(band(3.0)).toBe("mid");
    expect(band(1.5)).toBe("low");
    expect(band(null)).toBe("none");
  });
  it("formats time", () => {
    expect(duration(742)).toBe("12:22");
    expect(duration(3725)).toBe("1h 2m");
    expect(timestamp(65.9)).toBe("1:05");
  });
  it("badge text distinguishes unevaluated and partial", () => {
    expect(badgeText(null, false)).toBe("");
    expect(badgeText(null, true)).toBe("~");
    expect(badgeText(4.04, true)).toBe("4.0");
  });
});

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

describe("ApiClient", () => {
  beforeEach(async () => {
    window.localStorage.clear();
    await setApiBase("http://backend.test/");
  });

  it("creates an anonymous user once and sends the bearer token", async () => {
    const calls: [string, RequestInit | undefined][] = [];
    const fetchMock = vi.fn(async (url: string | URL | Request, init?: RequestInit) => {
      calls.push([String(url), init]);
      if (String(url).endsWith("/v1/users")) return jsonResponse(201, { user_id: "u1", token: "tok" });
      return jsonResponse(200, { request_id: "r", ranker_version: "linear_v1", items: [], notes: [] });
    });
    const client = new ApiClient(fetchMock as unknown as typeof fetch);
    await client.recommendations(5);
    await client.recommendations(5);
    expect(calls.filter(([u]) => u.endsWith("/v1/users"))).toHaveLength(1);
    const last = calls.at(-1)!;
    expect(last[0]).toBe("http://backend.test/v1/recommendations?limit=5");
    expect((last[1]?.headers as Record<string, string>).Authorization).toBe("Bearer tok");
  });

  it("shares one registration between concurrent first calls", async () => {
    let registrations = 0;
    const fetchMock = vi.fn(async (url: string | URL | Request) => {
      if (String(url).endsWith("/v1/users")) {
        registrations += 1;
        return jsonResponse(201, { user_id: "u1", token: "tok" });
      }
      return jsonResponse(200, {});
    });
    const client = new ApiClient(fetchMock as unknown as typeof fetch);
    await Promise.all([client.preferences(), client.recommendations(), client.preferences()]);
    expect(registrations).toBe(1);
  });

  it("surfaces API errors and drops a rejected token", async () => {
    window.localStorage.setItem("avd.token", JSON.stringify("stale"));
    const fetchMock = vi.fn(async () => jsonResponse(401, { detail: "invalid token" }));
    const client = new ApiClient(fetchMock as unknown as typeof fetch);
    await expect(client.preferences()).rejects.toEqual(new ApiError(401, "invalid token"));
    expect(window.localStorage.getItem("avd.token")).toBeNull();
  });
});

describe("modelLabel", () => {
  it("shows tag and quantization for local models and leaves hosted ids alone", () => {
    expect(modelLabel("ollama:qwen3.5:9b@0123456789ab:Q4_K_M")).toBe("qwen3.5:9b (Q4_K_M)");
    expect(modelLabel("claude-opus-5-5")).toBe("claude-opus-5-5");
  });
});
