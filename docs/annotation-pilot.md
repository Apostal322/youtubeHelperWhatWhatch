# Human annotation pilot: volunteers, no server

**Status (2026-10-20): no human rating exists.** The workflow below has been verified end to end
with scripted test input: `tests/test_annotation_offline.py`, and `scripts/e2e/annotation_packet.py`
in headless Chromium against the real pilot transcripts. **Data collection is pending:** it needs
real raters, and none have been recruited. No ratings were generated or simulated, and none will
be.

## Target

| | |
|---|---|
| Items | the 10 development episodes of `real_pilot_v1` (Changelog, CC BY-SA 4.0; all external uses confirmed) |
| Raters per item | 3, independent and blind |
| Calibration | 2 practice episodes per rater before any development item; never used as labels |
| Per dimension | a status (scored, insufficient evidence, not applicable, cannot determine, needs expertise, needs video/audio, ambiguous, abstain); a 1–5 score only when scored; confidence; citations the rater selects themselves |
| Separately | optional personal enjoyment (1–5), never used as a quality label |
| Never shown to raters | any automated assessment, other raters' work, view or subscriber counts, links to the video platform |

## Time each rater gives (an estimate, not a measurement)

The episodes are long conversational transcripts:

* the 10 development items are 9,062 to 13,768 words (109,420 in total);
* the 2 calibration items are 10,769 and 18,251 words.

Assuming careful reading at 200–250 words per minute, plus about 15 minutes to rate ten
dimensions with citations:

| Work | Estimate |
|---|---|
| One development item | 50–85 minutes |
| Calibration (both practice items) | 2.5–3 hours |
| All 10 development items | 10–12 hours |
| The whole pilot (30 ratings plus calibration for 3 raters) | about 40 rater-hours |

This is too much to ask of one volunteer, so spread it across more people. For example, **6 raters
each rating 5 items** gives every item 3 ratings, at about 6–7 hours plus calibration per person,
spread over a few weeks. Tell volunteers the honest figure. The first real packets will measure the
actual time: `avd annotate qc` reports reported durations.

## How it works without a central server

```
coordinator (has the data)                        rater (any computer, offline)
avd annotate create-rater <pseudonym>
avd annotate packet --rater <p> --items 3  ──►  packet-<p>.html (one file, opens in a browser,
                                                 no network access, progress saved locally)
                                           ◄──  ratings-<p>-xxxx.json (Download ratings file)
avd annotate import-packet ratings.json
avd annotate export --out benchmarks/labels/…
```

* **One file, no network.** The packet is a single HTML file with a content security policy that
  forbids every network request. The rater needs no account, no installation and no connection
  to anyone's server. The coordinator can send it by e-mail, a shared folder or a USB stick.
* **Blind assignment, as online:**
  * calibration first;
  * then the items with the fewest ratings, in a per-rater random order;
  * never more than 3 raters per item (assigned but unreturned items count toward the 3);
  * occasional blind repeats for test–retest, once a rater has 5 submissions.
* **Authenticated returns.** Each packet carries a random secret, stored hashed by the
  coordinator, and a return file is accepted only with it. Each item is validated on its own:
  * status rules;
  * every citation is checked against the transcript;
  * QC flags (`too_fast`, `straight_lining`, missing evidence).

  A rejected item stays assigned, so the rater can fix it and send the file again. Re-sending an
  accepted item records nothing twice.
* **Timing is self-reported.** Offline ratings carry `client_timing` (time the item was open and
  visible) and `offline_packet`, so analyses can separate them from server-timed work.
* **A rater who stops.** `avd annotate release-packet <id>` frees their unreturned items for
  someone else, and keeps whatever they already returned.

## Coordinator checklist

1. **Before recruiting:** approve the consent text below (it is a template; adapt it).
   Decide where returned files are kept and for how long.
2. `avd dataset import benchmarks/datasets/real_pilot_v1/manifest.json` on the coordinator's
   machine. Blinding of model assessments is on by default.
3. For each volunteer: `avd annotate create-rater <pseudonym> --expertise <self-declared tags>`.
   Ignore the printed token; packets do not need it.
4. `avd annotate packet --rater <pseudonym>`. The first packet holds the 2 calibration items.
   Send the file with the instructions below.
5. When a ratings file comes back: `avd annotate import-packet <file>`. Send the next packet with
   `avd annotate packet --rater <pseudonym> --items 2`, or `--items 3`.
6. After the calibration round, hold a short call, optional and voluntary, to discuss
   disagreements on the practice items. Do not discuss development items while they are being
   rated.
7. Weekly: `avd annotate qc`. When every development item has 3 ratings:
   `avd annotate export --out benchmarks/labels/real_pilot_v1-dev`, then
   `avd bench-real --labels benchmarks/labels/real_pilot_v1-dev`.

   **Human–human agreement is the first result.** It is reported before any comparison with a
   model, and with whatever value it has.
8. Do not change the rubric while ratings are being collected. Do not rate the test split until
   the development round is analysed.

## Invitation (template)

> **Help test whether AI can judge the quality of long-form talks: volunteers wanted**
>
> AI Video Discovery is an open-source research project. It tries to rate videos and podcasts on
> what they contain (depth, accuracy, clarity, structure, and so on) instead of how popular they
> are. Before anyone can trust such ratings, they have to be compared with careful human
> judgement, and right now we have none.
>
> We are looking for volunteers who are comfortable reading long English transcripts of
> technical podcasts (software, AI, product work). You would:
>
> * receive a single file that opens in your web browser; no account, no installation, no
>   internet connection needed while you work;
> * read two practice episodes, then a few real ones, and rate each on ten aspects using written
>   descriptions of each score. You can always say "the transcript doesn't show this" instead of
>   guessing;
> * send back the ratings file it produces.
>
> **Time:** each episode takes about 50–85 minutes to read and rate; the two practice episodes
> about 2.5–3 hours together. We ask for 5 episodes after practice, spread over a few weeks, at
> your own pace. This is unpaid volunteer work; you can stop at any time.
>
> You are identified only by a pseudonym you choose. Contact: <coordinator contact>.

## Participant instructions (template, send with the packet)

1. Open `packet-<you>.html` in Chrome, Firefox, Edge or Safari. It does not use the internet.
   Your progress is saved in that browser on that computer. Use the same browser each time, and
   do not clear its site data until you have sent your ratings.
2. Practice items are marked **[practice]**. Rate them like the others: they help everyone
   calibrate and are never counted.
3. For each of the ten aspects:
   * pick a **status** first. Choose **scored** only if you can justify a 1–5 score from the
     transcript;
   * read all five score descriptions before choosing;
   * select a few words in a transcript segment and press **Cite selection** to back your score.
     Mark a citation as *counter-evidence* if it points the other way.
4. Judge the **transcript only**: not the show, the speakers' reputation, popularity or
   production polish. Do not look up reviews or any automated rating of the episode. Each aspect
   is separate: a messy structure does not lower depth.
5. Unsure? Use *cannot determine*, *requires subject expertise* or *insufficient evidence*. An
   honest "can't tell" is more useful than a guess.
6. *Personal enjoyment* is optional and separate. It is never used as a quality rating.
7. Press **Mark this item complete** for each finished item, then **Download ratings file**, and
   send that file to the coordinator. You can send partial work and continue later.

## Consent (template, must be approved by the project owner before use)

> I agree to rate the transcripts in this packet for the AI Video Discovery research project. I
> understand that:
>
> * my ratings, notes and citations are stored under the pseudonym I chose, and my e-mail or name
>   is not stored with them;
> * the ratings may be published in aggregate (for example, agreement between raters), and in
>   pseudonymous form in the project's benchmark files;
> * the reported time I spent on each item is recorded;
> * I can withdraw at any time, and ask for my ratings to be excluded (they are then excluded
>   from all exports, with the reason recorded);
> * this is unpaid, voluntary work.

## What this does not provide

* Recruitment. The project has no budget and no rater pool.
* Verification of identity or expertise. Expertise tags are self-declared.
* Offline pairwise comparisons. Those still need the online annotation UI (`docs/annotation.md`).
* Protection against a rater who shares their packet. The secret authenticates the file, not
  the person, so treat it like a password.
