# Evaluation export format (`avd-evaluation-export`, version 1.0)

A portable, versioned file of assessments. Use it to compare installations or models, publish
research results, or back up evaluations without their source content.

```bash
avd evaluations export --out evals.json                     # current evaluations, no quotes
avd evaluations export --out evals.json --include-quotes    # quotes only where rights permit
avd evaluations export --out evals.json --no-synthetic --video VIDEO_ID --video ...
avd evaluations import evals.json --origin "alice's instance"   # validated, stored as unverified
```

## Contents

```jsonc
{
  "format": "avd-evaluation-export",
  "format_version": "1.0",            // readers accept any 1.x; 2.0 would be incompatible
  "created_at": "2026-10-19T12:00:00+00:00",
  "producer": {"software": "ai-video-discovery", "git_commit": "…"},
  "notice": "Assessments, not facts: …",
  "evaluations": [{
    "video": {"platform": "youtube", "video_id": "…", "title": "…", "is_synthetic": false},
    "evaluation_id": "…", "created_at": "…", "stage": "B",
    "mode": "llm",                     // llm | heuristic_baseline | synthetic_demo
    "evaluator_source": "local_ai",    // local_ai | hosted_ai | heuristic | synthetic
    "status": "complete",              // complete | partial | abstained | failed
    "framework_version": "2026.10.1",
    "versions": {"framework": "…", "rubric": "rubric_v2", "prompt": "…", …},
    "models": ["ollama:qwen3.5:4b@ab12cd34ef56:Q4_K_M"],   // served model with weights digest
    "config_hash": "…",                // evaluator configuration (incl. local model revision)
    "source_fingerprint": "…",         // hash of the exact evaluated input
    "genre": "education",
    "overall": {"score": 3.6, "confidence": 0.52, "abstain_reason": null, "method": "…"},
    "coverage": 0.71,
    "dimensions": [{
      "dimension": "explanation_clarity", "group": "content",
      "status": "scored",              // scored | not_applicable | insufficient_evidence | requires_media | failed
      "score": 4.0, "confidence": 0.6, "reason_code": null,
      "rubric_version": "rubric_v2", "method": "llm_rubric:…", "model": "ollama:…",
      "uncertainty": {…},
      "evidence": [{
        "kind": "supporting", "segment_index": 12, "start_seconds": 95.0, "end_seconds": 103.5,
        "quote_sha256": "…",           // always, when the evidence has a quote
        "quote": null,                 // text only when rights permit (below)
        "verified": true               // quote matched the transcript at that segment
      }]
    }],
    "validation": {"passed": true, "evidence_claimed": 14, "evidence_verified": 14, "ungrounded_dimensions": []},
    "rights": {
      "quotes_included": false, "quotes_reason": "quotes not requested",
      "transcript_included": false,
      "content_sources": [{"source_type": "licensed_dataset", "license": "CC-BY-SA-4.0", "content_sha256": "…"}]
    }
  }],
  "integrity": {"algorithm": "sha256", "evaluations_sha256": "…"}
}
```

Scores keep their meaning across installations: the same 1–5 anchored rubric, the same
abstention statuses and the same genre-profile aggregation, whatever the provider. A dimension
that could not be judged has no score; it is never exported as 0.

## What is never exported

* **Transcript text.** Not in any mode.
* **Quote text**, unless `--include-quotes` is given *and* every active source of that video
  permits public quotation:
  * project-authored synthetic fixtures: yes;
  * a licensed dataset whose quotation right is not pending review: yes;
  * creator-authorized captions and user-submitted content: **never**.

  Otherwise each quote is exported as its SHA-256. Anyone holding the same authorized
  transcript can check that the quote exists, and nobody else learns the text.
* **Personal data**: users, preferences, feedback, rater identities.
* **Evaluations of items under blind annotation**, which the public API also withholds.

Before publishing an export that includes quotes, check the licences' attribution
requirements: CC BY-SA requires attribution and share-alike.

## Integrity, authenticity and trust

* `integrity.evaluations_sha256` is the SHA-256 of the canonical JSON (sorted keys, no
  whitespace, UTF-8) of the `evaluations` array.
  * It detects accidental corruption and edits made after export.
  * It does **not** authenticate the producer: anyone can recompute it.
  * Trust a file because of where it came from (a signed release, a researcher you know), not
    because its hash matches.
* **Import is never authoritative.** An import:
  * validates the format (strict schema, size limits, value ranges, unknown fields refused);
  * checks the integrity hash, and checks every included quote against its own hash;
  * stores the evaluations in `imported_evaluations`, with the declared origin, as
    `imported_unverified`.

  Imported evaluations never replace local ones, never enter rankings or recommendations and
  never touch benchmark results. They are exposed read-only at
  `GET /v1/videos/{id}/imported-evaluations`. Importing the same file from the same origin
  twice is a no-op.
* **Signing** (e.g. minisign or Sigstore over the file) is the natural next step for
  authenticity. It is not implemented.

## Compatibility

* Readers accept major version 1; minor versions add optional fields only.
* A breaking change (renamed field, changed score meaning) requires `2.0` and a converter.

## Future: sharing between installations

Sharing is deliberately limited to files for now. Requirements before any automatic exchange:

* **Authenticity**: signed exports with published producer keys.
* **Compatibility**: refuse or flag mismatched rubric and framework versions. Scores from
  different rubrics are not comparable.
* **Conflicting scores**: show disagreement between models and instances as a result in itself.
  Never average it into a "consensus truth". Agreement between models is not validity.
* **Spam and reputation**: per-origin allow lists chosen by the operator; no global reputation
  without a moderation process.
* **Provenance**: model digest, prompt, rubric, config hash and source fingerprint are already in
  every record. Keep them mandatory.
* **Privacy**: exports carry no user data. Keep it that way.
* **Revocation**: an origin must be able to withdraw evaluations, e.g. when a source's rights
  are revoked. This needs a signed revocation list.
* **Moderation**: someone must be responsible for each shared feed. The project runs none.
