import { describe, expect, it } from "vitest";
import { type BadgeState, NOT_FOUND_TTL_MS, createBadgeUpdater } from "../src/lib/badge";

const A = "AAAAAAAAAAA";
const B = "BBBBBBBBBBB";
const url = (id: string) => `https://www.youtube.com/watch?v=${id}`;
const ok = (overall: number) =>
  new Response(JSON.stringify({ intrinsic_quality: { overall, synthetic: false } }), { status: 200 });

function harness(respond: (id: string) => Promise<Response>) {
  const shown = new Map<number, BadgeState>();
  let clock = 0;
  const calls: string[] = [];
  const updater = createBadgeUpdater({
    base: async () => "http://localhost:8000",
    allowed: async () => true,
    fetch: async (u) => {
      const id = u.split("/").pop() as string;
      calls.push(id);
      return respond(id);
    },
    set: async (tab, s) => void shown.set(tab, s),
    now: () => clock,
  });
  return { updater, shown, calls, tick: (ms: number) => (clock += ms) };
}

describe("toolbar badge", () => {
  it("a slow lookup for the previous video never overwrites the current one", async () => {
    let releaseA: (r: Response) => void = () => {};
    const h = harness((id) => (id === A ? new Promise((r) => (releaseA = r)) : Promise.resolve(ok(1.4))));
    const first = h.updater.refresh(1, url(A));
    await new Promise((r) => setTimeout(r, 0));
    await h.updater.refresh(1, url(B)); // YouTube changes video without a page load
    releaseA(ok(4.6));
    await first;
    expect(h.shown.get(1)?.text).toBe("1.4");
  });

  it("clears the old score as soon as the video changes", async () => {
    const shown: BadgeState[] = [];
    const updater = createBadgeUpdater({
      base: async () => "http://localhost:8000",
      allowed: async () => true,
      fetch: (u) => (u.endsWith(A) ? Promise.resolve(ok(4.6)) : new Promise(() => {})), // B never answers
      set: async (_t, s) => void shown.push(s),
      now: () => 0,
    });
    await updater.refresh(1, url(A));
    void updater.refresh(1, url(B));
    await new Promise((r) => setTimeout(r, 0));
    expect(shown.map((s) => s.text)).toEqual(["", "4.6", ""]);
    expect(shown.at(-1)?.title).toMatch(/checking/); // B shows "checking…", not A's 4.6
  });

  it("does not cache an unreachable or failing backend", async () => {
    let down = true;
    const h = harness(async () => {
      if (down) throw new TypeError("Failed to fetch");
      return ok(3.2);
    });
    await h.updater.refresh(1, url(A));
    expect(h.shown.get(1)?.title).toMatch(/unreachable/);
    down = false;
    await h.updater.refresh(1, url(A));
    expect(h.shown.get(1)?.text).toBe("3.2");
  });

  it("reports a server error as an error, not as 'not analyzed'", async () => {
    const h = harness(async () => new Response("", { status: 503 }));
    await h.updater.refresh(1, url(A));
    expect(h.shown.get(1)).toEqual({ text: "!", title: expect.stringMatching(/backend error \(HTTP 503\)/) });
  });

  it("re-checks 'not analyzed yet' soon, so a finished evaluation appears", async () => {
    let evaluated = false;
    const h = harness(async () => (evaluated ? ok(3.9) : new Response("", { status: 404 })));
    await h.updater.refresh(1, url(A));
    evaluated = true;
    await h.updater.refresh(1, url(A));
    expect(h.shown.get(1)?.text).toBe(""); // still within the short negative cache
    h.tick(NOT_FOUND_TTL_MS + 1);
    await h.updater.refresh(1, url(A));
    expect(h.shown.get(1)?.text).toBe("3.9");
  });

  it("resets badge and title on pages that are not a video", async () => {
    const h = harness(async () => ok(4.6));
    await h.updater.refresh(1, url(A));
    await h.updater.refresh(1, "https://www.youtube.com/feed/subscriptions");
    expect(h.shown.get(1)).toEqual({ text: "", title: "AI Video Discovery" });
  });
});
