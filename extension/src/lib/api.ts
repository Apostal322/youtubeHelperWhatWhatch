import { getItem, removeItem, setItem } from "./storage";
import type {
  Claim,
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

export const REQUEST_TIMEOUT_MS = 15_000;

/** A failed API call. ``code`` follows the backend error taxonomy (``not_found``,
 * ``rate_limited``, ``validation_error`` ...); network failures and timeouts use ``network`` and
 * ``timeout``. ``requestId`` lets a user quote the failing request in a bug report. */
export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public code: string = "error",
    public requestId: string | null = null,
    public retryAfterSeconds: number | null = null,
  ) {
    super(message);
  }
}

export function friendlyMessage(e: unknown): string {
  if (!(e instanceof ApiError)) return e instanceof Error ? e.message : String(e);
  switch (e.code) {
    case "rate_limited":
      return `Too many requests; try again in ${e.retryAfterSeconds ?? 10}s.`;
    case "network":
      return "Cannot reach the backend. Check that it is running and the API base URL in Preferences.";
    case "timeout":
      return "The backend took too long to respond. Try again.";
    case "unavailable":
      return `The backend is temporarily unavailable: ${e.message}`;
    default:
      return e.message;
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
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    let res: Response;
    try {
      res = await this.fetchImpl(`${base}${path}`, {
        ...init,
        signal: controller.signal,
        headers: { ...headers, ...(init.headers ?? {}) },
      });
    } catch (err) {
      const aborted = err instanceof DOMException && err.name === "AbortError";
      throw new ApiError(0, aborted ? "request timed out" : "network error", aborted ? "timeout" : "network");
    } finally {
      clearTimeout(timer);
    }
    if (!res.ok) {
      let detail = res.statusText;
      let code = res.status === 429 ? "rate_limited" : "error";
      let requestId = res.headers.get("X-Request-ID");
      try {
        const body = (await res.json()) as {
          detail?: unknown;
          error?: { code?: string; message?: string; request_id?: string | null };
        };
        detail = body.error?.message ?? (typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body));
        code = body.error?.code ?? code;
        requestId = body.error?.request_id ?? requestId;
      } catch {
        /* non-JSON error */
      }
      if (res.status === 401 && auth) await removeItem(TOKEN_KEY);
      const retry = Number(res.headers.get("Retry-After"));
      throw new ApiError(res.status, detail, code, requestId, Number.isFinite(retry) && retry > 0 ? retry : null);
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
  claims(id: string): Promise<{ video_id: string; items: Claim[] }> {
    return this.request(`/v1/videos/${encodeURIComponent(id)}/claims`);
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
