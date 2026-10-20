import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { BackendSettings } from "../src/popup/components/Backend";
import { hasRemoteConsent } from "../src/lib/backend";

const status = {
  deployment_mode: "server",
  registration: "invite",
  inference: { location: "local", provider: "ollama", model: "qwen3.5:9b", reachable: true, model_installed: true },
  database: "ok",
  queue: { queued: 0 },
  versions: { framework: "2026.10.1" },
};

describe("BackendSettings", () => {
  beforeEach(() => {
    window.localStorage.clear();
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(JSON.stringify(status), { status: 200, headers: { "Content-Type": "application/json" } })),
    );
  });
  afterEach(() => vi.unstubAllGlobals());

  it("refuses plain HTTP to another computer", async () => {
    render(<BackendSettings />);
    fireEvent.change(screen.getByLabelText(/Backend address/), { target: { value: "http://avd.example.org" } });
    fireEvent.click(screen.getByText("Connect"));
    expect((await screen.findByRole("alert")).textContent).toMatch(/Remote backends must use HTTPS/);
  });

  it("asks for explicit consent before using a remote server, after showing its diagnostics", async () => {
    render(<BackendSettings />);
    fireEvent.change(screen.getByLabelText(/Backend address/), { target: { value: "https://avd.example.org" } });
    fireEvent.click(screen.getByText("Connect"));
    const go = (await screen.findByText("Connect to avd.example.org")) as HTMLButtonElement;
    expect(screen.getByText(/Its operator will receive what you send it/)).toBeTruthy();
    expect(screen.getAllByText(/shared server/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/qwen3\.5:9b · model ready/).length).toBeGreaterThan(0);
    expect(go.disabled).toBe(true);
    expect(await hasRemoteConsent("https://avd.example.org")).toBe(false);
    fireEvent.click(screen.getByLabelText(/I understand and want to connect/));
    expect(go.disabled).toBe(false);
  });

  it("shows a clear status when the local backend is down", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => Promise.reject(new TypeError("Failed to fetch"))));
    render(<BackendSettings />);
    await waitFor(() => expect(screen.getByText(/Not connected/)).toBeTruthy());
    expect(screen.getByText(/avd start/, { selector: "div" })).toBeTruthy();
  });
});
