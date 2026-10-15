import { getItem, removeItem, setItem } from "./storage";
import type {
  Evaluation,
  FeedbackBody,
  Meta,
  Preferences,
  QualitySummary,
  RecommendationResponse,
  SearchItem,
  VideoCard,
} from "./types";

export const DEFAULT_API_BASE = "http://localhost:8000";
const BASE_KEY = "avd.apiBase";
const TOKEN_KEY = "avd.token";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

export async function getApiBase(): Promise<string> {
  return (await getItem<string>(BASE_KEY)) ?? DEFAULT_API_BASE;
}

export async function setApiBase(url: string): Promise<void> {
  await setItem(BASE_KEY, url.replace(/\/+$/, ""));
  await removeItem(TOKEN_KEY); // tokens are per backend
}

type FetchLike = typeof fetch;

export class ApiClient {
  private pendingToken: Promise<string> | null = null;

  constructor(private fetchImpl: FetchLike = (...a) => fetch(...a)) {}

  private async request<T>(path: string, init: RequestInit = {}, auth = false): Promise<T> {
    const base = await getApiBase();
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (auth) headers.Authorization = `Bearer ${await this.token()}`;
    const res = await this.fetchImpl(`${base}${path}`, { ...init, headers: { ...headers, ...(init.headers ?? {}) } });
    if (!res.ok) {
      let detail = res.statusText;
      try {
        const body = (await res.json()) as { detail?: unknown };
        detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
      } catch {
        /* non-JSON error */
      }
      if (res.status === 401 && auth) await removeItem(TOKEN_KEY);
      throw new ApiError(res.status, detail);
    }
    return (await res.json()) as T;
  }

  /** Anonymous account created lazily on first authenticated call. Concurrent callers share one
   * registration request instead of each creating an account. */
  async token(): Promise<string> {
    const existing = await getItem<string>(TOKEN_KEY);
    if (existing) return existing;
    this.pendingToken ??= this.request<{ user_id: string; token: string }>("/v1/users", { method: "POST" })
      .then(async (created) => {
        await setItem(TOKEN_KEY, created.token);
        return created.token;
      })
      .finally(() => {
        this.pendingToken = null;
      });
    return this.pendingToken;
  }

  meta(): Promise<Meta> {
    return this.request("/v1/meta");
  }
  recommendations(limit = 20): Promise<RecommendationResponse> {
    return this.request(`/v1/recommendations?limit=${limit}`, {}, true);
  }
  video(id: string): Promise<{ video: VideoCard; intrinsic_quality: QualitySummary | null; available_modalities: string[] }> {
    return this.request(`/v1/videos/${encodeURIComponent(id)}`);
  }
  evaluation(id: string): Promise<Evaluation> {
    return this.request(`/v1/videos/${encodeURIComponent(id)}/evaluation`);
  }
  search(q: string, opts: { minQuality?: number; genre?: string } = {}): Promise<{ items: SearchItem[] }> {
    const p = new URLSearchParams({ q });
    if (opts.minQuality) p.set("min_quality", String(opts.minQuality));
    if (opts.genre) p.set("genre", opts.genre);
    return this.request(`/v1/search?${p.toString()}`);
  }
  preferences(): Promise<Preferences> {
    return this.request("/v1/users/me/preferences", {}, true);
  }
  savePreferences(p: Preferences): Promise<Preferences> {
    return this.request("/v1/users/me/preferences", { method: "PUT", body: JSON.stringify(p) }, true);
  }
  feedback(body: FeedbackBody): Promise<{ id: number }> {
    return this.request("/v1/feedback", { method: "POST", body: JSON.stringify(body) }, true);
  }
  submitTranscript(
    videoId: string,
    body: { format: "vtt" | "srt" | "text"; content: string; language?: string; attestation: string },
  ): Promise<{ created: boolean; status: string }> {
    return this.request(
      `/v1/videos/${encodeURIComponent(videoId)}/transcripts`,
      { method: "POST", body: JSON.stringify(body) },
      true,
    );
  }
}

export const api = new ApiClient();
