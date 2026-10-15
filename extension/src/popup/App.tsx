import { useEffect, useState } from "react";
import { api } from "../lib/api";
import type { Meta } from "../lib/types";
import { parseVideoId } from "../lib/youtube";
import { DiscoverView, PreferencesView, SearchView, VideoView } from "./components/Views";

type Tab = "discover" | "video" | "search" | "prefs";
const TABS: { id: Tab; label: string }[] = [
  { id: "discover", label: "Discover" },
  { id: "video", label: "Video quality" },
  { id: "search", label: "Search" },
  { id: "prefs", label: "Preferences" },
];

async function activeTabVideoId(): Promise<string | null> {
  if (typeof chrome === "undefined" || !chrome.tabs) return null;
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return parseVideoId(tab?.url);
}

export function App() {
  const [tab, setTab] = useState<Tab>("discover");
  const [videoId, setVideoId] = useState<string | null>(null);
  const [meta, setMeta] = useState<Meta | null>(null);
  const [offline, setOffline] = useState(false);
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    api.meta().then(setMeta).catch(() => setOffline(true));
    void activeTabVideoId().then((id) => {
      if (id) {
        setVideoId(id);
        setTab("video");
      }
    });
  }, []);

  const select = (id: string) => {
    setVideoId(id);
    setTab("video");
  };

  return (
    <div className="app">
      <header>
        <h1>AI Video Discovery</h1>
        {meta ? (
          <span className={meta.demo_mode ? "mode mode-demo" : "mode"} title={meta.disclaimer}>
            {meta.demo_mode ? "Synthetic demo" : `LLM · ${meta.llm_provider}`}
          </span>
        ) : null}
      </header>
      <nav role="tablist">
        {TABS.map((t) => (
          <button key={t.id} type="button" role="tab" aria-selected={tab === t.id} className={tab === t.id ? "active" : ""} onClick={() => setTab(t.id)}>
            {t.label}
          </button>
        ))}
      </nav>
      <main>
        {offline && tab !== "prefs" ? (
          <div className="error">
            Cannot reach the backend. Start it (see README) or set its URL under Preferences.
          </div>
        ) : null}
        {tab === "discover" ? <DiscoverView key={reloadKey} onSelect={select} /> : null}
        {tab === "video" ? <VideoView videoId={videoId} /> : null}
        {tab === "search" ? <SearchView onSelect={select} /> : null}
        {tab === "prefs" ? <PreferencesView onSaved={() => setReloadKey((k) => k + 1)} /> : null}
      </main>
    </div>
  );
}
