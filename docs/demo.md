# Demo: how the application works, in ten minutes, on your own computer

This walkthrough uses only content you are allowed to use: the project's **synthetic demo
videos** (written for the project; they do not exist on YouTube) and **a transcript you write
yourself**. Nothing is downloaded from YouTube, no model is needed and no server of the project
is involved.

> **What you see is not AI and not research evidence.** Without a model, every assessment comes
> from the **heuristic baseline** (keyword and structure features). Each one says so:
> `evaluator_source` is `synthetic` or `heuristic`, and `disclaimer` states that it is not an AI
> assessment and not validated against people. The scores show *how the system presents a
> judgement*, not how good any video is.

Verified on Linux (2026-10-20, `main`). Windows and macOS: not tested.

## 1. Start a local instance with the demo catalog

```bash
cd backend && source .venv/bin/activate          # after the install in README.md
avd start --demo --port 8000                     # http://127.0.0.1:8000, this computer only
```

It is ready in about 2 seconds, and the embedded worker evaluates the 20 synthetic videos
within seconds. `curl -s localhost:8000/v1/status` shows the worker, queue counts, catalog size
and `"evaluation_mode": "heuristic_baseline"`.

## 2. Submit content: a transcript you wrote

```bash
TOKEN=$(curl -s -X POST localhost:8000/v1/users | python3 -c "import sys,json;print(json.load(sys.stdin)['token'])")
curl -s -X POST localhost:8000/v1/videos/own_bread_001/transcripts \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"format":"text","language":"en",
       "attestation":"I wrote this text myself and have the right to submit it.",
       "content":"Today we look at why bread rises. Yeast eats sugars in the dough and releases carbon dioxide. ..."}'
# {"content_source_id": "...", "created": true, "status": "queued for evaluation"}   (HTTP 202)
```

The attestation is stored with the source (`authorization_basis`, `license: user-attested`).
You can delete it, and everything derived from it, with `DELETE /v1/users/me`.

## 3. Job progress

Until the evaluation exists, `GET /v1/videos/own_bread_001/evaluation` answers 404 with the
video's state, for example `no evaluation yet (status: content_available)`. `GET /v1/status`
shows the queue (`queued`, `running`, `succeeded`) and the worker's current job. With the
heuristic baseline a short transcript is done within a second. A local model takes minutes.

## 4. The quality breakdown, evidence and uncertainty

```bash
curl -s localhost:8000/v1/videos/own_bread_001/evaluation
```

What came back for the bread transcript (heuristic baseline):

| Dimension | Status | Score | Evidence |
|---|---|---:|---:|
| information density | scored | 3.9 | 1 verified quote |
| depth | scored | 1.7 | 1 |
| factual accuracy | insufficient evidence (`method_cannot_judge`) | — | 0 |
| originality | insufficient evidence (`method_cannot_judge`) | — | 0 |
| clarity | scored | 3.3 | 1 |
| title consistency | insufficient evidence (`insufficient_material`: no title) | — | 0 |
| audio quality, visual quality, editing, visual readability | requires media (`missing_modality`) | — | 0 |

* **Overall** 2.79 with **confidence 0.08** and coverage 0.58. The technical group was left out
  and the weights renormalised, and the response says so in `overall.notes`.
* Missing modalities are **abstentions, not zeros**: no audio or video, so no score.
* Evidence entries carry the segment index, the timestamps and `verified: true` (the quote was
  found at that place in the transcript).
* The baseline also scores some dimensions without a quote (pacing, structure, storytelling).
  That is a limitation of the heuristic. The AI path requires evidence or an abstention.

## 5. Search and recommendations

```bash
curl -s "localhost:8000/v1/search?q=water%20density"      # 1 result: the synthetic science video about ice
curl -s "localhost:8000/v1/search?q=quantum%20chromodynamics"   # {"items": []}: nothing matches
curl -s -X PUT localhost:8000/v1/users/me/preferences -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" -d '{"interests":["physics","chemistry"]}'
curl -s localhost:8000/v1/recommendations -H "Authorization: Bearer $TOKEN"
```

* Every recommendation explains itself, for example `Matches your interest "physics"`,
  `Strongest: pacing 4.9/5`, `Assessment: synthetic demo; confidence 0.35 (not calibrated
  against people yet)`.
* `notes` states how many candidates matched (`4 of 21 analyzed candidates match your interests
  closely enough; the list is not filled up with unrelated videos`). Off-topic **exploration
  picks** are labelled as such.
* **Known limit:** the default `hashing` embedder is a keyword-hash fallback. Its collisions can
  rank an unrelated video near the top: for "bread yeast", a synthetic Spanish video on study
  technique scored 0.223, just above the bread transcript at 0.214. The MiniLM embedder
  (`AVD_EMBEDDING_PROVIDER=onnx_minilm`, after `avd embeddings fetch-model`) separated them
  clearly: 0.54 against 0.05.

## 6. The same in the browser extension

Build and load the extension (README), open its popup and go to **Discover**, **Search** and
**This video**. These show the cards, explanations and the quality breakdown with evidence and
timestamps. **Preferences → Backend** shows the connection, the evaluator ("heuristic baseline
(no AI model configured)") and the catalog counts. The end-to-end script
`scripts/e2e/full_local.py` drives exactly this, in Chromium, and saves screenshots.

## 7. With a real local model

`avd start --provider ollama` after `avd local doctor` and `avd local pull …`
([`local-ai.md`](local-ai.md)). If the model server is not running, `avd start` prints
`evaluator: local ollama …: NOT READY (Ollama unreachable …)`. Jobs then wait in the queue and
are retried; they are never silently evaluated by the heuristic. **Preferences → Backend** in
the extension says `model server unreachable`.
