// Which backend the extension talks to, and whether the user agreed to talk to it.
//
// There is no default remote server and no directory of instances: the extension talks to a
// backend on this computer unless the user types another address and explicitly agrees to send
// it their requests. Remote backends must use HTTPS; plain HTTP is accepted only for loopback.

import { getItem, setItem } from "./storage";

export type BackendKind = "local" | "remote";

export interface BackendCheck {
  ok: boolean;
  /** Normalized base URL (origin only, no trailing slash). */
  base?: string;
  kind?: BackendKind;
  error?: string;
}

const LOOPBACK = new Set(["localhost", "127.0.0.1", "[::1]"]);

/** Validate a user-typed backend URL. Only an origin is accepted: no credentials, query,
 * fragment or path (the API lives at the root), and HTTPS for anything not on this computer. */
export function validateBackendUrl(input: string): BackendCheck {
  let u: URL;
  try {
    u = new URL(input.trim());
  } catch {
    return { ok: false, error: "Not a valid URL (example: http://localhost:8000 or https://avd.example.org)." };
  }
  if (u.protocol !== "http:" && u.protocol !== "https:") return { ok: false, error: "Use http:// or https://." };
  if (u.username || u.password) return { ok: false, error: "Do not put credentials in the URL." };
  if ((u.pathname && u.pathname !== "/") || u.search || u.hash)
    return { ok: false, error: "Give only the server address, without a path or query." };
  const local = LOOPBACK.has(u.hostname);
  if (!local && u.protocol !== "https:")
    return { ok: false, error: "Remote backends must use HTTPS (plain HTTP is only allowed for localhost)." };
  return { ok: true, base: u.origin, kind: local ? "local" : "remote" };
}

const CONSENT_KEY = "avd.remoteConsent";
const BADGE_KEY = "avd.remoteBadgeLookups";

/** Origins the user explicitly agreed to send requests to (interests, feedback, viewed videos). */
export async function hasRemoteConsent(origin: string): Promise<boolean> {
  return ((await getItem<string[]>(CONSENT_KEY)) ?? []).includes(origin);
}

export async function grantRemoteConsent(origin: string): Promise<void> {
  const list = (await getItem<string[]>(CONSENT_KEY)) ?? [];
  if (!list.includes(origin)) await setItem(CONSENT_KEY, [...list, origin]);
}

/** Whether the toolbar badge may look up every YouTube video you open on a *remote* backend.
 * Off by default: that would tell the server operator what you watch. Local backends always may. */
export async function remoteBadgeLookups(): Promise<boolean> {
  return (await getItem<boolean>(BADGE_KEY)) ?? false;
}

export async function setRemoteBadgeLookups(on: boolean): Promise<void> {
  await setItem(BADGE_KEY, on);
}

export function isLocalBase(base: string): boolean {
  try {
    return LOOPBACK.has(new URL(base).hostname);
  } catch {
    return false;
  }
}

export interface Diagnosis {
  reachable: boolean;
  latencyMs?: number;
  deploymentMode?: string;
  registration?: string;
  authRequiredForRead?: boolean;
  inference?: { location: string | null; provider: string; model?: string; reachable?: boolean; model_installed?: boolean; error?: string };
  worker?: Record<string, unknown>;
  database?: string;
  queue?: Record<string, number>;
  catalog?: { videos: number; evaluated: number; synthetic: number; hidden_for_blind_annotation?: number };
  framework?: string;
  problem?: string;
}

/** Catalog size in words, so an empty catalog reads as "nothing imported yet", not as broken. */
export function catalogLine(d: Diagnosis): string {
  const c = d.catalog;
  if (!c) return "Catalog: unknown (older backend)";
  if (c.videos === 0) return "Catalog: empty. Nothing has been imported yet (see Discover for how to start).";
  const real = c.videos - c.synthetic;
  const working = (d.queue?.queued ?? 0) + (d.queue?.running ?? 0);
  const hidden = c.hidden_for_blind_annotation ?? 0;
  return `Catalog: ${c.videos} video${c.videos === 1 ? "" : "s"} (${real} real, ${c.synthetic} synthetic demo) · ${c.evaluated} evaluated${
    working ? ` · ${working} job${working === 1 ? "" : "s"} in progress` : ""
  }${hidden ? ` · ${hidden} hidden during blind annotation` : ""}`;
}

/** Connection diagnostics against ``/v1/status`` (no credentials sent). */
export async function diagnose(base: string, fetchImpl: typeof fetch = (...a) => fetch(...a), timeoutMs = 8000): Promise<Diagnosis> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  const t0 = Date.now();
  try {
    const res = await fetchImpl(`${base}/v1/status?check_inference=true`, { signal: controller.signal });
    const latencyMs = Date.now() - t0;
    if (!res.ok) {
      let code = "";
      try {
        code = ((await res.json()) as { error?: { code?: string } }).error?.code ?? "";
      } catch {
        /* not JSON */
      }
      const problem =
        code === "invalid_host"
          ? "The server rejected this host name. Its operator must add it to AVD_ALLOWED_HOSTS."
          : code === "local_only"
            ? "That backend is a personal (local-only) instance and refuses other computers."
            : `The server answered HTTP ${res.status}.`;
      return { reachable: true, latencyMs, problem };
    }
    const s = (await res.json()) as Record<string, unknown>;
    const versions = (s.versions ?? {}) as Record<string, string>;
    return {
      reachable: true,
      latencyMs,
      deploymentMode: s.deployment_mode as string | undefined,
      registration: s.registration as string | undefined,
      authRequiredForRead: s.auth_required_for_read as boolean | undefined,
      inference: s.inference as Diagnosis["inference"],
      worker: s.worker as Record<string, unknown> | undefined,
      database: s.database as string | undefined,
      queue: s.queue as Record<string, number> | undefined,
      catalog: s.catalog as Diagnosis["catalog"],
      framework: versions.framework,
    };
  } catch (e) {
    const aborted = e instanceof DOMException && e.name === "AbortError";
    return {
      reachable: false,
      problem: aborted
        ? "No answer within 8 seconds."
        : "Cannot connect. Is the backend running (avd start), and is the address right? For HTTPS, check the certificate.",
    };
  } finally {
    clearTimeout(timer);
  }
}
