# real_pilot_v1: datasheet

Real, licensed, long-form spoken content for **human-grounded** validation of transcript-based
quality assessment. This is the first non-synthetic dataset in the project.

**It contains no quality labels.** Labels can only come from human raters through the blind
annotation workflow (`docs/annotation.md`). Until people submit annotations, every
"human-agreement" metric for this dataset is *not available*. Nothing in this repository fills
that gap with model output or authored labels.

| | |
|---|---|
| Items | 29 episodes from 10 shows (2 publishers) |
| Words | 344,328 in total; per episode median 11,904 (range 8,905–18,251) |
| Length | median 67 min of transcript (49–98 min); 20 items over 60 min |
| Language | English (identified per item; all `en`) |
| Splits (by show) | calibration 2 · dev 10 · test 17 |
| Manifest | `manifest.json`, items digest `5bc30338ce62614125be0bcf42df89d3a1f7238bede318a9472d11c30eaf92dc` |
| Selection seed | 20261015 |

## Motivation

The synthetic dataset (`synthetic_v1`) can only show that the pipeline behaves as designed. It
cannot show that scores match human judgement. This pilot provides real transcripts with clear
rights so that people can annotate them and model–human agreement can be measured. 29 items is
a pilot, sized for checking feasibility and inter-rater agreement. It is not a statistically
powered validation set.

## Composition

| Group (split) | Publisher / licence | Items | Topic area |
|---|---|---|---|
| `changelog:friends` (calibration) | Changelog Media, CC BY-SA 4.0 | 2 | tech culture (casual conversation) |
| `changelog:backstage` (dev) | Changelog Media, CC BY-SA 4.0 | 2 | media production |
| `changelog:founderstalk` (dev) | Changelog Media, CC BY-SA 4.0 | 2 | business / entrepreneurship |
| `changelog:gotime` (dev) | Changelog Media, CC BY-SA 4.0 | 3 | Go programming |
| `changelog:practicalai` (dev) | Changelog Media, CC BY-SA 4.0 | 3 | AI / machine learning |
| `changelog:brainscience` (test) | Changelog Media, CC BY-SA 4.0 | 2 | psychology / neuroscience |
| `changelog:jsparty` (test) | Changelog Media, CC BY-SA 4.0 | 3 | web / JavaScript |
| `changelog:podcast` (test) | Changelog Media, CC BY-SA 4.0 | 3 | software engineering |
| `changelog:shipit` (test) | Changelog Media, CC BY-SA 4.0 | 2 | infrastructure / operations |
| `talkpython` (test) | Talk Python To Me, GPL-3.0 | 7 | Python programming (YouTube editions) |

Each item has: source path and pinned commit, raw and normalised checksums, provenance URL,
derived metadata (show, episode, title and title source), word, segment and duration counts,
timing granularity, normalisation counts, validation issues, split, and stratum.

## Collection and selection

1. Sources are the publishers' own repositories, fetched at pinned commits:
   `thechangelog/transcripts@44a0e16c` and `mikeckennedy/talk-python-transcripts@accd851f`.
   No media is downloaded, and nothing is scraped from YouTube or third-party mirrors.
2. Per group: the sorted file list was shuffled with `random.Random("<seed>:<source>:<group>")`,
   and the first eligible files were kept up to the quota. Eligibility: English, 3k–20k words,
   and for Changelog at least 15 source timestamps and at least 1 per 1,000 words. Every
   rejection and its reason is recorded in `manifest.json → selection.per_group`.
3. Whole groups (shows) were assigned to splits with a seeded greedy fill (target 10/30/60 %). No
   show appears in two splits.

Exclusions: Changelog `bigtent/` (Grafana Labs copyright, not CC), `news/` and `afk/` (no
timestamps), `scripts/`. Talk Python: only `youtube_transcripts/` (the video editions).

Shortfall: `changelog:brainscience` yielded 2 of its quota of 3, because only 2 of its 34 episodes
have at least 15 timestamps. The pilot therefore has 29 items, not 30.

## Preprocessing

* **Changelog** (Markdown with `**Speaker:**` paragraphs and inline `\[HH:MM:SS\]` timestamps
  about once a minute). Each speaker paragraph becomes one segment. A segment carries the
  *source span* it falls in: from the last timestamp at or before it to the next one. Times are
  never interpolated, so several paragraphs share one span (median 11 per span; range 7–21 across items). Citations are therefore
  exact at about one-minute granularity, not per sentence. The text before the first timestamp
  starts at 0:00. The last span's end is unknown and is set to its start. Ad-break labels are
  dropped, but their timestamps are kept as anchors.
* **Talk Python** (WebVTT with 1–5 s cues). Cues are merged into segments of about 20 s (at most
  45 s), keeping each segment's first-cue start and last-cue end.
* Ingestion normalisation (`prep_v2`): whitespace, empty segments, duplicate cues. Shared source
  spans are kept as they are.

## Rights

Recorded per content source at import: owner, attribution, licence, provenance URL, retrieval
time, raw checksum, permitted processing, storage terms, restrictions and revocation policy.

| | Changelog | Talk Python |
|---|---|---|
| Licence | CC BY-SA 4.0 | GPL-3.0 (+ README: "Feel free to use the files to help you explore the conversations!") |
| Permitted here | store, analyse, quote short excerpts with attribution, derive metrics | same |
| Restrictions | attribution; adaptations of the text must be CC BY-SA | redistributed text stays GPL-3.0 |
| In this repo | manifest only (paths, checksums, metadata); transcripts are fetched at import, not redistributed | same |
| Open legal question | Are evaluations "analysis" rather than adaptations? Needs counsel review. | Is GPL appropriate for prose? Analysis use needs counsel review. |

Attribution: transcripts © Changelog Media LLC (CC BY-SA 4.0) and © Talk Python To Me / Michael
Kennedy (GPL-3.0). The annotation UI and exports show these attributions.

**Revocation.** `avd dataset revoke <video_id> --reason ...` marks the content sources revoked,
withdraws current evaluations (history is kept for audit), and stops further annotation
assignment of the item.

## Uses

* Intended: blind human annotation; model–human and human–human agreement; evidence-grounding
  checks on real long-form transcripts; stress tests for long-transcript processing (chunking,
  map-reduce); false-positive tests for the injection detector on benign content.
* Not intended: ranking these shows or creators; training production models; any claim about
  YouTube content in general.

## Known limitations

* **Genre homogeneity.** Every item is an interview or discussion podcast, and most are about
  software. The pilot says nothing about lectures, tutorials, reviews, comedy, storytelling,
  music or vlogs. Genre-specific aggregation cannot be validated with it.
* **English only.**
* **Coarse timestamps** (Changelog), about one minute. Evidence localisation is limited to that.
* **Transcription method.** Changelog transcripts are human-edited. The Talk Python files do not
  document how they were produced. Transcript errors may affect clarity judgements.
* **Titles.** Changelog titles are unavailable offline and are derived ("Go Time #226"), so
  title consistency is not assessable for those items. Talk Python titles are the creator's
  file slugs (including the source's own misspelling "syntatic").
* **Split shift.** All Talk Python (WebVTT, cue timing) items fall in the test split. Dev has
  only Changelog-format items.
* **Annotation cost.** Each item takes about 45–75 minutes to read carefully. With 3 raters per
  dev/test item, that is about 81 annotations, or roughly 60–100 rater-hours.
* **Small n.** Confidence intervals will be wide. The bootstrap must resample by group (show).

## Reproduce

```bash
git clone --filter=blob:none https://github.com/thechangelog/transcripts changelog
git clone --filter=blob:none https://github.com/mikeckennedy/talk-python-transcripts talkpython
avd dataset build-pilot --clone changelog=changelog --clone talkpython=talkpython   # same items digest
avd dataset import benchmarks/datasets/real_pilot_v1/manifest.json --evaluate          # verifies checksums
```

`import` fails closed on a changed upstream byte (raw checksum), on parser drift (normalised
content hash), on a commit mismatch, or on missing licence or authorisation fields.
