// MV3 service worker: shows the backend's overall score for the current YouTube video as the
// toolbar badge. It only *reads* precomputed evaluations from the backend (cached for 10 minutes);
// it never runs model inference.

import { getApiBase } from "./lib/api";
import { hasRemoteConsent, isLocalBase, remoteBadgeLookups } from "./lib/backend";
import { badgeText } from "./lib/format";
import { parseVideoId } from "./lib/youtube";

const TTL_MS = 10 * 60 * 1000;
const cache = new Map<string, { at: number; text: string; title: string }>();

async function lookup(videoId: string): Promise<{ text: string; title: string }> {
  const hit = cache.get(videoId);
  if (hit && Date.now() - hit.at < TTL_MS) return hit;
  let text = "";
  let title = "AI Video Discovery — not analyzed yet";
  const base = await getApiBase();
  // Looking up every video you open tells the backend what you watch. That is fine on your own
  // computer; for a remote backend it needs both the connection consent and a separate opt-in.
  if (!isLocalBase(base) && !((await hasRemoteConsent(base)) && (await remoteBadgeLookups()))) {
    return { text: "", title: "AI Video Discovery — badge lookups off for remote backends (see Preferences)" };
  }
  try {
    const res = await fetch(`${base}/v1/videos/${encodeURIComponent(videoId)}`);
    if (res.ok) {
      const body = (await res.json()) as { intrinsic_quality: { overall: number | null; synthetic: boolean } | null };
      const q = body.intrinsic_quality;
      text = badgeText(q?.overall, q !== null);
      title = q
        ? `AI Video Discovery — overall ${q.overall?.toFixed(1) ?? "n/a (see breakdown)"}${q.synthetic ? " (synthetic demo)" : ""}`
        : "AI Video Discovery — in catalog, not evaluated yet";
    }
  } catch {
    title = "AI Video Discovery — backend unreachable";
  }
  const entry = { at: Date.now(), text, title };
  cache.set(videoId, entry);
  return entry;
}

async function refresh(tabId: number, url: string | undefined): Promise<void> {
  const id = parseVideoId(url);
  if (!id) {
    await chrome.action.setBadgeText({ tabId, text: "" });
    return;
  }
  const { text, title } = await lookup(id);
  await chrome.action.setBadgeBackgroundColor({ tabId, color: "#3b5bdb" });
  await chrome.action.setBadgeText({ tabId, text });
  await chrome.action.setTitle({ tabId, title });
}

chrome.tabs.onUpdated.addListener((tabId, info, tab) => {
  if (info.url || info.status === "complete") void refresh(tabId, tab.url);
});
chrome.tabs.onActivated.addListener(({ tabId }) => {
  chrome.tabs.get(tabId, (tab) => void refresh(tabId, tab.url));
});
