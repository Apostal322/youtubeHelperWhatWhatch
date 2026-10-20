# DRAFT: invitation to contributors

> **DRAFT FOR PROJECT-OWNER APPROVAL. Do not publish or send.** The project owner decides
> whether, where and when to post this. It promises nothing the project cannot provide: there
> is no payment, no hosting and no guaranteed response time.

---

**AI Video Discovery: open-source, popularity-blind assessment of long-form videos and podcasts**

AI Video Discovery tries to judge talks, lectures and podcasts by what they contain (depth,
accuracy, clarity, structure, and so on) instead of views or likes. Every judgement cites the
transcript or abstains. It is Apache-2.0 software that anyone runs on their own computer or
server; the project operates no service.

**Where it stands, honestly.** The software works: a FastAPI backend with SQLite or
PostgreSQL, a Chrome extension, local models through Ollama, offline rating packets and a
benchmark harness, all with an extensive test suite. What it does **not** have yet is evidence:
**no real model has evaluated anything** (zero real model calls), and **no human has rated
anything**. Without a model, every score comes from a heuristic baseline, labelled as such
everywhere. Contributions that produce or check evidence are worth the most.

**Ways to help** (setup takes about five minutes; no GPU or API key needed for development):

* **Local AI contributors.** If your computer can run a 4B model (16 GB RAM recommended), one
  run of one openly licensed transcript is the project's first real AI evidence:
  [`docs/first-experiment.md`](../first-experiment.md), Part A. The results stay
  *contributor-reported* until someone reproduces them.
* **ML and evaluation researchers.**
  * Review the rubric, the evidence and abstention rules, and the pre-registered release gates
    ([`docs/evaluation.md`](../evaluation.md), [`docs/research-contributing.md`](../research-contributing.md)).
  * Criticise the relevance benchmark's design (`benchmarks/datasets/relevance_v1/README.md`).
  * Check whether the agreement statistics are the right ones.
* **Python developers.** See [`docs/good-first-issues.md`](../good-first-issues.md): search
  precision with the default embedder, PostgreSQL restore, CLI JSON output. Ground rules are in
  [`CONTRIBUTING.md`](../../CONTRIBUTING.md): no simulated output presented as real, no
  popularity signals in quality scores, no tuning on held-out data.
* **Browser-extension developers.** Imported-evaluation display, dark-mode contrast, and
  testing on browsers other than Chromium.
* **Windows and macOS users.** The install instructions have never been run on either. A report
  of what breaks is a real contribution: use the *Setup help* issue form.

Human rating volunteers have a separate call, with its own consent process.

Questions: <the repository's issue tracker, or the owner's chosen contact>.
