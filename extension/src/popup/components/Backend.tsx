import { useEffect, useState } from "react";
import { DEFAULT_API_BASE, getApiBase, setApiBase, setInviteCode } from "../../lib/api";
import {
  type Diagnosis,
  diagnose,
  grantRemoteConsent,
  hasRemoteConsent,
  remoteBadgeLookups,
  setRemoteBadgeLookups,
  validateBackendUrl,
} from "../../lib/backend";

/** Connection status and diagnostics for the current backend, shown as text. */
export function ConnectionStatus({ base, d }: { base: string; d: Diagnosis | null }) {
  if (!d) return <div className="muted small">Checking {base}…</div>;
  if (!d.reachable || d.problem)
    return (
      <div className="error small" role="alert">
        <strong>Not connected</strong> to {base}. {d.problem}
      </div>
    );
  const inf = d.inference;
  const evaluator =
    !inf || inf.provider === "heuristic"
      ? "heuristic baseline (no AI model configured)"
      : `${inf.location === "local" ? "local" : "hosted"} ${inf.provider} · ${inf.model ?? "?"}${
          inf.location === "local"
            ? inf.reachable === false
              ? ` · model server unreachable (${inf.error ?? "error"})`
              : inf.model_installed === false
                ? " · model not installed"
                : inf.model_installed
                  ? " · model ready"
                  : ""
            : ""
        }`;
  return (
    <div className="small" role="status">
      <div>
        <strong>Connected</strong> to {base} ({d.latencyMs} ms) · {d.deploymentMode === "server" ? "shared server" : "personal instance"}
        {d.framework ? ` · framework ${d.framework}` : ""}
      </div>
      <div>Evaluator: {evaluator}</div>
      <div>
        Database: {d.database ?? "?"} · queued jobs: {d.queue?.queued ?? 0}
        {d.registration && d.registration !== "open" ? ` · registration: ${d.registration}` : ""}
      </div>
    </div>
  );
}

export function BackendSettings() {
  const [current, setCurrent] = useState(DEFAULT_API_BASE);
  const [input, setInput] = useState(DEFAULT_API_BASE);
  const [diag, setDiag] = useState<Diagnosis | null>(null);
  const [pending, setPending] = useState<{ base: string; d: Diagnosis } | null>(null);
  const [agree, setAgree] = useState(false);
  const [invite, setInvite] = useState("");
  const [badge, setBadge] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void getApiBase().then((b) => {
      setCurrent(b);
      setInput(b);
      void diagnose(b).then(setDiag);
    });
    void remoteBadgeLookups().then(setBadge);
  }, []);

  const apply = async (base: string) => {
    if (typeof chrome !== "undefined" && chrome.permissions) {
      const granted = await chrome.permissions.request({ origins: [`${base}/*`] });
      if (!granted) return setError("Chrome permission to contact that backend was not granted.");
    }
    await setInviteCode(invite);
    await setApiBase(base);
    window.location.reload();
  };

  const check = async () => {
    setError(null);
    setPending(null);
    const v = validateBackendUrl(input);
    if (!v.ok || !v.base) return setError(v.error ?? "Invalid URL");
    if (v.kind === "local" || (await hasRemoteConsent(v.base))) return apply(v.base);
    // Remote: diagnose first (only /v1/status, no personal data), then ask for explicit consent.
    setPending({ base: v.base, d: await diagnose(v.base) });
  };

  return (
    <fieldset>
      <legend>Backend</legend>
      <ConnectionStatus base={current} d={diag} />
      <label>
        Backend address
        <input value={input} onChange={(e) => setInput(e.target.value)} placeholder="http://localhost:8000" />
      </label>
      <div className="muted small">
        Default: a backend on this computer (<code>avd start</code>). A remote backend must use HTTPS and is
        operated by someone else, never by this project.
      </div>
      <label>
        Invite code (only if the server requires one)
        <input value={invite} onChange={(e) => setInvite(e.target.value)} autoComplete="off" />
      </label>
      <button type="button" onClick={() => void check()}>
        Connect
      </button>
      <button type="button" className="link" onClick={() => void diagnose(current).then(setDiag)}>
        Re-check connection
      </button>
      {error ? (
        <div className="error small" role="alert">
          {error}
        </div>
      ) : null}
      {pending ? (
        <div className="consent" role="dialog" aria-label="Connect to a remote backend">
          <ConnectionStatus base={pending.base} d={pending.d} />
          <p className="small">
            <strong>{new URL(pending.base).host}</strong> is not on this computer. Its operator will receive what you
            send it: your interests and preferences, feedback, searches and the videos whose assessments you open. Only
            connect to servers you trust.
          </p>
          <label className="check">
            <input type="checkbox" checked={agree} onChange={(e) => setAgree(e.target.checked)} />I understand and want
            to connect to this server
          </label>
          <button
            type="button"
            disabled={!agree || !pending.d.reachable}
            onClick={() => void grantRemoteConsent(pending.base).then(() => apply(pending.base))}
          >
            Connect to {new URL(pending.base).host}
          </button>
        </div>
      ) : null}
      <label className="check">
        <input
          type="checkbox"
          checked={badge}
          onChange={(e) => {
            setBadge(e.target.checked);
            void setRemoteBadgeLookups(e.target.checked);
          }}
        />
        Show the toolbar badge on YouTube for a remote backend (sends each video you open to that server)
      </label>
    </fieldset>
  );
}
