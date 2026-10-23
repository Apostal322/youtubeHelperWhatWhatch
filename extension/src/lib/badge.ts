// Toolbar badge for the current YouTube video, kept correct across YouTube's in-page (history)
// navigation: lookups for an earlier video that finish late are dropped, the badge is cleared as
// soon as the video changes, and only real answers are cached.

import { badgeText } from "./format";
import { parseVideoId } from "./youtube";

export const TTL_MS = 10 * 60 * 1000; // an evaluation rarely changes; re-checked after 10 minutes
export const NOT_FOUND_TTL_MS = 30 * 1000; // "not analyzed yet" may change as soon as a job finishes
export const DEFAULT_TITLE = "AI Video Discovery";

export interface BadgeState {
  text: string;
  title: string;
}

export interface BadgeDeps {
  base(): Promise<string>;
  allowed(base: string): Promise<boolean>; // badge lookups permitted for this backend
  fetch(url: string): Promise<Response>;
  set(tabId: number, state: BadgeState): Promise<void>;
  now(): number;
}

type Entry = BadgeState & { at: number; ttl: number };

export function createBadgeUpdater(deps: BadgeDeps) {
  const cache = new Map<string, Entry>();
  const latest = new Map<number, string | null>(); // what each tab shows now

  async function lookup(base: string, videoId: string): Promise<BadgeState> {
    const key = `${base}|${videoId}`;
    const hit = cache.get(key);
    if (hit && deps.now() - hit.at < hit.ttl) return hit;
    let res: Response;
    try {
      res = await deps.fetch(`${base}/v1/videos/${encodeURIComponent(videoId)}`);
    } catch {
      return { text: "", title: `${DEFAULT_TITLE} — backend unreachable` }; // not cached: retried next time
    }
    if (res.status === 404) {
      const entry = { text: "", title: `${DEFAULT_TITLE} — not analyzed yet`, at: deps.now(), ttl: NOT_FOUND_TTL_MS };
      cache.set(key, entry);
      return entry;
    }
    if (!res.ok) return { text: "!", title: `${DEFAULT_TITLE} — backend error (HTTP ${res.status})` };
    const body = (await res.json()) as {
      intrinsic_quality: { overall: number | null; synthetic: boolean } | null;
    };
    const q = body.intrinsic_quality;
    const state = {
      text: badgeText(q?.overall, q !== null),
      title: q
        ? `${DEFAULT_TITLE} — overall ${q.overall?.toFixed(1) ?? "n/a (see breakdown)"}${q.synthetic ? " (synthetic demo)" : ""}`
        : `${DEFAULT_TITLE} — in catalog, not evaluated yet`,
    };
    cache.set(key, { ...state, at: deps.now(), ttl: q ? TTL_MS : NOT_FOUND_TTL_MS });
    return state;
  }

  async function refresh(tabId: number, url: string | undefined): Promise<void> {
    const id = parseVideoId(url);
    const changed = latest.get(tabId) !== id;
    latest.set(tabId, id);
    if (!id) {
      await deps.set(tabId, { text: "", title: DEFAULT_TITLE });
      return;
    }
    // Never leave the previous video's score on screen while this one is looked up.
    if (changed) await deps.set(tabId, { text: "", title: `${DEFAULT_TITLE} — checking…` });
    const base = await deps.base();
    // Looking up every video you open tells the backend what you watch. That is fine on your own
    // computer; for a remote backend it needs both the connection consent and a separate opt-in.
    const state = (await deps.allowed(base))
      ? await lookup(base, id)
      : { text: "", title: `${DEFAULT_TITLE} — badge lookups off for remote backends (see Preferences)` };
    if (latest.get(tabId) !== id) return; // the tab moved on to another video meanwhile
    await deps.set(tabId, state);
  }

  return { refresh, forget: (tabId: number) => latest.delete(tabId) };
}
