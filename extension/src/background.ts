// MV3 service worker: shows the backend's overall score for the current YouTube video as the
// toolbar badge. It only *reads* precomputed evaluations from the backend; it never runs model
// inference. The logic (and why it is safe across YouTube's in-page navigation) is in lib/badge.

import { getApiBase } from "./lib/api";
import { createBadgeUpdater } from "./lib/badge";
import { hasRemoteConsent, isLocalBase, remoteBadgeLookups } from "./lib/backend";

const badge = createBadgeUpdater({
  base: getApiBase,
  allowed: async (base) => isLocalBase(base) || ((await hasRemoteConsent(base)) && (await remoteBadgeLookups())),
  fetch: (url) => fetch(url),
  now: () => Date.now(),
  set: async (tabId, { text, title }) => {
    try {
      await chrome.action.setBadgeBackgroundColor({ tabId, color: "#3b5bdb" });
      await chrome.action.setBadgeText({ tabId, text });
      await chrome.action.setTitle({ tabId, title });
    } catch {
      // the tab was closed while the lookup ran
    }
  },
});

chrome.tabs.onUpdated.addListener((tabId, info, tab) => {
  if (info.url || info.status === "complete") void badge.refresh(tabId, tab.url);
});
chrome.tabs.onActivated.addListener(({ tabId }) => {
  chrome.tabs.get(tabId, (tab) => {
    if (!chrome.runtime.lastError) void badge.refresh(tabId, tab.url);
  });
});
chrome.tabs.onRemoved.addListener((tabId) => badge.forget(tabId));
