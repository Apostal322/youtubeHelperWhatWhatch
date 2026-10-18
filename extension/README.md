# AI Video Discovery — Chrome extension (Manifest V3)

React + TypeScript + Vite. Talks only to your AI Video Discovery backend; it never runs model
inference itself.

## Build and load

```bash
npm ci
npm run build        # → dist/
npm test             # vitest (URL parsing, formatting, API client and errors, breakdown, claims, states)
npm run typecheck
```

Chrome → `chrome://extensions` → *Developer mode* → *Load unpacked* → `dist/`.

The backend defaults to `http://localhost:8000`. To use another backend, set its URL under
*Preferences → Backend*; Chrome will ask for permission to contact that origin.

## Features

| Tab | What it shows |
|---|---|
| **Discover** | Personalised feed. Each card shows the intrinsic quality pill with its assessment state, factual reasons it was recommended (matched interest, strongest dimensions, genre weight and where it came from, confidence), and a *Why?* table with every ranking feature's contribution. Filters: *Evidence-checked only*; an eligibility note says how many items were excluded and why. Feedback: 👍 useful, 👎 not useful, *Not interested*. |
| **Video quality** | Opens automatically on YouTube watch pages. Assessment state, a plain-language summary (strongest/weakest dimensions and what could not be judged, and why), overall (genre profile, confidence, coverage), group scores, every dimension with status, justification, verified evidence quotes with timestamps — counter-evidence is marked as such — missing inputs, validation notes, *Claims and fact checks* (claim type, verdict, risk domain, cited reference passages and licences), and *Report an incorrect evaluation*. Videos without an evaluation offer authorized transcript submission. |
| **Search** | Topic search over analyzed videos with an optional minimum quality. |
| **Preferences** | Intent (learn, stay informed, be entertained, deep dive), interests, depth, languages, minimum/maximum duration, dimension priority multipliers, genre weights (v2 genres inherit their parent's weight unless set), diversity, minimum quality, optional small-creator boost, synthetic-content toggle, backend URL. |

The toolbar badge shows the backend's overall score for the YouTube video in the active tab (`~`
when only a partial evaluation exists). Synthetic demo evaluations are always labelled as such.

### Assessment states

Every score is shown with one of these labels, in text (not colour alone):

| State | Meaning |
|---|---|
| Evidence-checked | AI judgements whose cited quotes were verified against the transcript; validation passed, no review flags. **Not yet validated against human raters.** |
| Preliminary | Scores exist with caveats (review flags, unverified or missing evidence, partial coverage, low confidence); the caveats are listed. |
| Insufficient evidence | Too little could be judged from the authorized material. |
| Evaluation failed | No scores are shown. |
| Heuristic baseline | Keyword/structure statistics on real content; not an AI or human assessment. The header reads *Heuristic mode (not AI)* when no AI evaluator is configured. |
| Synthetic demo | Synthetic fixture content; never a real rating. |

### Reliability and accessibility

* Requests time out after 15 s; errors distinguish *backend unreachable*, *timeout*, *rate limited*
  (with the server's `Retry-After`) and server errors, show the request id for support, and offer
  *Retry*.
* Claim and quote text from transcripts is untrusted and rendered as text only.
* Visible keyboard focus, `role="status"` on loading indicators, labelled meters and notes, and
  `prefers-reduced-motion` support.

Screenshots (heuristic mode, licensed pilot data): [`docs/screenshots`](../docs/screenshots).

## Permissions

* `storage` — settings and the anonymous token.
* Host access to `www.youtube.com` — read the current video id from the tab URL.
* Host access to `localhost:8000` — the default backend; other backends are requested at runtime
  (`optional_host_permissions`).
