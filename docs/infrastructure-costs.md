# Who pays for what

AI Video Discovery is software, not a service. The project does not operate any server and does
not need one to work. Nothing in the code depends on infrastructure run, or paid for, by the
project's maintainers.

| | Personal computer (`avd start`) | Self-hosted server (Docker Compose) | Community instance (future, optional) |
|---|---|---|---|
| API, database, worker | you (your computer) | the server's operator | that instance's operator |
| AI inference | you: local Ollama, or a provider you configure and pay | the operator: local model or their provider account | that operator |
| Storage of transcripts, evaluations, embeddings | you | the operator | that operator |
| User accounts | none needed (single user) | the operator's database | that operator |
| Bandwidth | you | the operator | that operator |
| YouTube Data API quota (optional discovery) | your Google Cloud project and key | the operator's key | that operator's key |
| **The project maintainers** | **nothing** | **nothing** | **nothing** |

What the project provides, at no running cost to anyone:

* source code, issues and reviews on GitHub;
* CI on GitHub Actions for pull requests (GitHub's free tier for public repositories; CI never
  needs a GPU, a model or an API key);
* documentation in the repository.

What the project does **not** provide, and the software does not assume:

* a public API, catalog, model endpoint, account system or update server;
* telemetry. There is none: `AVD_TELEMETRY` exists only so that nothing can ever be turned on
  silently, and nothing reads it to send data;
* hosted demos. A free tier is never treated as permanent infrastructure.

## Outbound connections an instance makes

Only those its operator configures:

| Destination | When | Setting |
|---|---|---|
| Local Ollama (`localhost:11434` or a private host) | AI evaluation with `AVD_LLM_PROVIDER=ollama` | `AVD_OLLAMA_BASE_URL` (local/private only unless `AVD_OLLAMA_ALLOW_REMOTE=true`) |
| A hosted model API (Anthropic, an OpenAI-compatible URL) | only with that provider configured and its key set | `AVD_LLM_PROVIDER`, `AVD_*_API_KEY`, `AVD_OPENAI_BASE_URL` |
| `www.googleapis.com` (YouTube Data API) | discovery and metadata, only with `AVD_YOUTUBE_API_KEY` | `AVD_YOUTUBE_API_KEY` |
| Licensed dataset repositories (pinned commits) | `avd dataset ...` commands run by the operator | — |
| Hugging Face (pinned MiniLM embedding model) | only `avd embeddings fetch-model`, run by the operator | — |
| Ollama's model registry | only `avd local pull` / `ollama pull`, with size approval | — |

There is no fallback from a local model to a paid API. If the configured provider fails, jobs
wait and retry; they are never rerouted.

## If someone runs a community instance

They can, under the project license. They pay for and are responsible for it, including its
privacy policy, rights compliance and moderation. The extension never connects to an instance
by default, has no built-in list of instances, and asks for explicit consent before it sends
anything to a server on another computer.
