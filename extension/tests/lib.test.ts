import { beforeEach, describe, expect, it, vi } from "vitest";
import { ApiClient, ApiError, setApiBase, setInviteCode } from "../src/lib/api";
import { diagnose, grantRemoteConsent, hasRemoteConsent, isLocalBase, validateBackendUrl } from "../src/lib/backend";
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
    await grantRemoteConsent("http://backend.test");
  });

  it("sends nothing to a remote backend before the user agrees", async () => {
    window.localStorage.clear();
    await setApiBase("https://avd.example.org");
    const fetchMock = vi.fn();
    const client = new ApiClient(fetchMock as unknown as typeof fetch);
    await expect(client.meta()).rejects.toMatchObject({ code: "consent_required" });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("talks to a local backend without a consent step", async () => {
    window.localStorage.clear();
    await setApiBase("http://localhost:8000");
    const fetchMock = vi.fn(async () => jsonResponse(200, { framework_version: "x" }));
    await new ApiClient(fetchMock as unknown as typeof fetch).meta();
    expect(fetchMock).toHaveBeenCalledOnce();
  });

  it("sends the invite code only when registering", async () => {
    await setInviteCode("join-us");
    const calls: [string, RequestInit | undefined][] = [];
    const fetchMock = vi.fn(async (url: string | URL | Request, init?: RequestInit) => {
      calls.push([String(url), init]);
      if (String(url).endsWith("/v1/users")) return jsonResponse(201, { user_id: "u1", token: "tok" });
      return jsonResponse(200, { interests: [] });
    });
    await new ApiClient(fetchMock as unknown as typeof fetch).preferences();
    const reg = calls.find(([u]) => u.endsWith("/v1/users"))!;
    expect((reg[1]!.headers as Record<string, string>)["X-Invite-Code"]).toBe("join-us");
    const prefs = calls.find(([u]) => u.endsWith("/preferences"))!;
    expect((prefs[1]!.headers as Record<string, string>)["X-Invite-Code"]).toBeUndefined();
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

  it("re-registers once and retries when the server no longer knows the token", async () => {
    window.localStorage.setItem("avd.token", JSON.stringify("stale"));
    const seen: string[] = [];
    const fetchMock = vi.fn(async (url: string | URL | Request, init?: RequestInit) => {
      const auth = (init?.headers as Record<string, string>)?.Authorization ?? "";
      seen.push(`${String(url).replace("http://backend.test", "")} ${auth}`);
      if (String(url).endsWith("/v1/users")) return jsonResponse(201, { user_id: "u2", token: "fresh" });
      return auth === "Bearer fresh" ? jsonResponse(200, { interests: ["x"] }) : jsonResponse(401, { detail: "invalid token" });
    });
    const prefs = await new ApiClient(fetchMock as unknown as typeof fetch).preferences();
    expect(prefs).toEqual({ interests: ["x"] });
    expect(seen).toEqual([
      "/v1/users/me/preferences Bearer stale",
      "/v1/users ",
      "/v1/users/me/preferences Bearer fresh",
    ]);
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

describe("backend selection", () => {
  it("accepts local HTTP and remote HTTPS origins only", () => {
    expect(validateBackendUrl("http://localhost:8000/")).toEqual({ ok: true, base: "http://localhost:8000", kind: "local" });
    expect(validateBackendUrl("http://127.0.0.1:9000").kind).toBe("local");
    expect(validateBackendUrl("http://[::1]:8000").kind).toBe("local");
    expect(validateBackendUrl("https://avd.example.org").kind).toBe("remote");
    for (const bad of [
      "http://avd.example.org",
      "http://192.168.1.10:8000",
      "ftp://x.org",
      "https://user:pw@avd.example.org",
      "https://avd.example.org/api",
      "https://avd.example.org/?x=1",
      "javascript:alert(1)",
      "not a url",
    ]) {
      expect(validateBackendUrl(bad).ok, bad).toBe(false);
    }
  });

  it("remembers consent per origin", async () => {
    window.localStorage.clear();
    expect(await hasRemoteConsent("https://a.example")).toBe(false);
    await grantRemoteConsent("https://a.example");
    expect(await hasRemoteConsent("https://a.example")).toBe(true);
    expect(await hasRemoteConsent("https://b.example")).toBe(false);
    expect(isLocalBase("http://localhost:8000") && !isLocalBase("https://a.example")).toBe(true);
  });

  it("diagnoses a healthy backend, a rejected host and an unreachable one", async () => {
    const ok = await diagnose("http://localhost:8000", (async () =>
      jsonResponse(200, {
        deployment_mode: "local",
        registration: "open",
        inference: { location: "local", provider: "ollama", model: "qwen3.5:4b", reachable: true, model_installed: false },
        database: "ok",
        queue: { queued: 2 },
        versions: { framework: "2026.10.1" },
      })) as unknown as typeof fetch);
    expect(ok).toMatchObject({ reachable: true, deploymentMode: "local", framework: "2026.10.1" });
    expect(ok.inference?.model_installed).toBe(false);
    const host = await diagnose("https://a.example", (async () =>
      jsonResponse(400, { error: { code: "invalid_host" } })) as unknown as typeof fetch);
    expect(host.problem).toMatch(/AVD_ALLOWED_HOSTS/);
    const down = await diagnose("http://localhost:1", (async () => {
      throw new TypeError("Failed to fetch");
    }) as unknown as typeof fetch);
    expect(down.reachable).toBe(false);
  });
});
