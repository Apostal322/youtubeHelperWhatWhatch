# DRAFT: call for model-benchmark volunteers

> **DRAFT FOR PROJECT-OWNER APPROVAL. Do not publish or send.** The project owner decides
> whether, where and when to post this, and fills in the placeholders in angle brackets.

---

**Run one small AI model on your own computer for an open research project**

AI Video Discovery is an open-source project that tries to assess long-form videos and podcasts
by what they contain (depth, accuracy, clarity, structure and similar) instead of how popular
they are. So far, no real AI model has evaluated anything for the project: its development
environment cannot download models. One run on your machine would be its first real evidence.

**What you would do**

* Install Git, Python 3.12 and [Ollama](https://ollama.com/download), clone the repository and
  download one small open model (about 2–3 GB).
* Run one command that evaluates **one** public podcast transcript: Practical AI #178,
  published by Changelog Media under CC BY-SA 4.0.
* Check the result with two more commands, and, if you want to, attach the resulting
  `smoke.zip` to a GitHub issue.

The copy-paste steps are in
[`docs/first-experiment.md`](../first-experiment.md), Part A.

**What you need**

* A computer with 16 GB of RAM (8 GB works, slowly) and 10 GB of free disk.
* No GPU is required. **We cannot promise a speed:** nobody has measured it yet. On a CPU, one
  evaluation of a 9,000-word transcript may take many minutes. Measuring that is part of the
  point.
* About 1–3 hours, mostly unattended.
* A free GitHub account, only if you submit.

**What it costs and what it does not need**

* Nothing. The model runs on your own computer. There is no paid API, no API key, no account
  with us, and no project server.
* Ollama stays on your own machine (`127.0.0.1`). Please do not expose it to your network.

**What you would share, if you choose to submit**

`smoke.zip` contains scores, counts, timings, the model's per-dimension output for that one
openly licensed transcript, the model name and digest, the Ollama version, the prompt and rubric
versions, the git commit, and a hardware summary (operating system, CPU model, core count, RAM,
GPU model).

It does **not** contain your name, user name, computer name, IP address or file paths. Before
writing the file, `avd community pack` checks for transcript text beyond the cited quotes,
credentials, e-mail addresses, network addresses, local paths and symbolic links, and refuses if
it finds any. You can inspect everything first with `avd community inspect`.

**Never upload** the `smoke-run` folder itself (its database contains transcript text), any
API key or token, or anything else from your computer.

**What your result will and will not mean**

It is recorded as *contributor-reported*: it shows whether the model produced valid output,
whether its quotes exist in the transcript, and how fast it ran on your hardware. It does not show
whether the model's quality judgments are right. That needs independent human ratings, which
are collected separately. A result with imperfect numbers is just as useful as one with good
numbers. Please do not rerun until the numbers look better.

**Platforms.** The steps have been tested on Linux up to the model call. The Windows and macOS
steps have **not** been tested. If anything differs, please open a *Setup help* issue.

Questions: <coordinator contact or issue link>.
