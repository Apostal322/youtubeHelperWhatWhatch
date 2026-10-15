# AI Video Discovery — Chrome extension (Manifest V3)

React + TypeScript + Vite. Talks only to your AI Video Discovery backend; it never runs model
inference itself.

## Build and load

```bash
npm ci
npm run build        # → dist/
npm test             # vitest (URL parsing, formatting, API client, breakdown rendering)
npm run typecheck
```

Chrome → `chrome://extensions` → *Developer mode* → *Load unpacked* → `dist/`.

The backend defaults to `http://localhost:8000`. To use another backend, set its URL under
*Preferences → Backend*; Chrome will ask for permission to contact that origin.

## Features

| Tab | What it shows |
|---|---|
| **Discover** | Personalised feed. Each card shows the intrinsic quality pill (or *partial* when the overall score abstains), why it was recommended, and a *Why?* table with every ranking feature's contribution. Feedback: 👍 useful, 👎 not useful, *Not interested*. |
| **Video quality** | Opens automatically on YouTube watch pages. Overall (genre profile, confidence, coverage), group scores, every dimension with status, justification, verified evidence quotes with timestamps (linked to the video when it is real), missing inputs, validation notes, and *Report an incorrect evaluation*. Videos without an evaluation offer authorized transcript submission. |
| **Search** | Topic search over analyzed videos with an optional minimum quality. |
| **Preferences** | Interests, depth, languages, dimension priority multipliers, genre weights, diversity, minimum quality, optional small-creator boost, synthetic-content toggle, backend URL. |

The toolbar badge shows the backend's overall score for the YouTube video in the active tab (`~`
when only a partial evaluation exists). Synthetic demo evaluations are always labelled as such.

## Permissions

* `storage` — settings and the anonymous token.
* Host access to `www.youtube.com` — read the current video id from the tab URL.
* Host access to `localhost:8000` — the default backend; other backends are requested at runtime
  (`optional_host_permissions`).
