// chrome.storage.local in the extension; localStorage when running the popup with `vite dev`.

const hasChromeStorage = (): boolean =>
  typeof chrome !== "undefined" && typeof chrome.storage !== "undefined" && !!chrome.storage.local;

export async function getItem<T>(key: string): Promise<T | undefined> {
  if (hasChromeStorage()) {
    const res = await chrome.storage.local.get(key);
    return res[key] as T | undefined;
  }
  try {
    const raw = window.localStorage.getItem(key);
    return raw === null ? undefined : (JSON.parse(raw) as T);
  } catch {
    return undefined;
  }
}

export async function setItem<T>(key: string, value: T): Promise<void> {
  if (hasChromeStorage()) {
    await chrome.storage.local.set({ [key]: value });
    return;
  }
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* storage unavailable (private mode): settings simply won't persist */
  }
}

export async function removeItem(key: string): Promise<void> {
  if (hasChromeStorage()) {
    await chrome.storage.local.remove(key);
    return;
  }
  try {
    window.localStorage.removeItem(key);
  } catch {
    /* ignore */
  }
}
