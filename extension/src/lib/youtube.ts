// Extract a YouTube video id from the URLs YouTube uses for watch pages.
const ID = /^[A-Za-z0-9_-]{11}$/;

export function parseVideoId(raw: string | undefined | null): string | null {
  if (!raw) return null;
  let url: URL;
  try {
    url = new URL(raw);
  } catch {
    return null;
  }
  const host = url.hostname.replace(/^www\.|^m\./, "");
  let id: string | null = null;
  if (host === "youtube.com" || host === "music.youtube.com") {
    if (url.pathname === "/watch") id = url.searchParams.get("v");
    else {
      const m = url.pathname.match(/^\/(?:shorts|live|embed)\/([^/?#]+)/);
      id = m?.[1] ?? null;
    }
  } else if (host === "youtu.be") {
    id = url.pathname.slice(1).split("/")[0] ?? null;
  }
  return id && ID.test(id) ? id : null;
}
