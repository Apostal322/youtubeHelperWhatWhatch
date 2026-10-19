# Self-hosting

Run AI Video Discovery on a machine you control, for yourself or for a group. Nothing in this
setup contacts a project-operated service; there is none. Costs and responsibilities:
[`infrastructure-costs.md`](infrastructure-costs.md).

**Status:**

* The Compose file is validated with `docker compose config` in CI, and CI builds the image.
* The full stack and the lite profile were **not** brought up with Docker in the development
  environment (no Docker daemon there).
* `avd start` (the lite profile's command) was run end to end outside Docker. See the Phase 3.2
  report.

## Choose a shape

| | Lite | Full stack |
|---|---|---|
| Command | `docker compose --profile lite up -d --build lite` | `docker compose up -d --build` |
| Services | 1 container | PostgreSQL+pgvector, Redis, API, Celery worker + beat |
| Database | SQLite in volume `litedata` | PostgreSQL in volume `pgdata` |
| Good for | one person, or a few on a home server | many users, continuous processing, more throughput |
| Local AI | add `--profile local-ai` (both) | |

Both use the same evaluation, aggregation and recommendation code. A score means the same
thing in either.

## Personal use with Docker (local mode)

```bash
git clone https://github.com/Apostal322/youtubeHelperWhatWhatch && cd youtubeHelperWhatWhatch
docker compose --profile lite up -d --build lite              # http://127.0.0.1:8000
docker compose --profile lite run --rm lite avd seed-demo      # optional synthetic examples
```

The API is published on `127.0.0.1` only. In local mode it accepts requests from loopback and
from the Docker bridge (`AVD_LOCAL_PEER_NETWORKS`). Nothing else on the network can use it.

## A shared server (server mode)

1. **`.env`** next to `docker-compose.yml` (never commit it):

   ```dotenv
   AVD_DEPLOYMENT_MODE=server
   AVD_ALLOWED_HOSTS=["avd.example.org"]
   AVD_ADMIN_TOKEN=<at least 24 random characters, e.g. `openssl rand -hex 24`>
   AVD_USER_REGISTRATION=invite              # invite | closed | open
   AVD_INVITE_CODES=<code-1>,<code-2>        # give these to your users
   AVD_REQUIRE_AUTH_FOR_READ=false           # true = private instance
   POSTGRES_PASSWORD=<random>
   AVD_FORWARDED_ALLOW_IPS=172.16.0.0/12     # where your reverse proxy connects from
   # AI provider: see docs/model-configuration.md and docs/local-ai.md
   AVD_LLM_PROVIDER=ollama
   ```

   The API **refuses to start** in server mode if any of these is unsafe:
   * a missing or short admin token, or the `change-me` placeholder;
   * allowed hosts that are only loopback names;
   * invite registration without codes;
   * public metrics;
   * the default database password.

   Run `avd doctor` to see the problems first.
2. **Start:** `docker compose --profile local-ai up -d --build`. The API runs its migrations at
   start.
3. **TLS reverse proxy.** The API port is on `127.0.0.1:8000` and the proxy terminates HTTPS.
   For example, Caddy on the host:

   ```caddyfile
   avd.example.org {
       reverse_proxy 127.0.0.1:8000
   }
   ```

   Set `AVD_FORWARDED_ALLOW_IPS` to the address the proxy connects from, so rate limits apply
   per client and not to the proxy as a whole.
4. **Users** connect the extension to `https://avd.example.org`, confirm the connection, and
   enter an invite code.

Never expose ports 5432 (PostgreSQL), 6379 (Redis) or 11434 (Ollama). The Compose file does not
publish the first two, and binds Ollama to `127.0.0.1`.

## Resource recommendations

| | Measured | Recommendation |
|---|---|---|
| API + embedded worker, heuristic evaluator | ~120 MB RSS, 11 MB data for 49 items (Linux VM) | 1 vCPU, 512 MB RAM |
| Full stack, heuristic evaluator | not measured | 2 vCPU, 2 GB RAM (PostgreSQL, Redis, 3 Python processes) |
| Local AI | not measured: no model could be installed in the dev environment | `avd local doctor` on the target machine. CPU-only: a 4B model needs ~3.5 GB more RAM (estimate) and evaluates slowly |

## Backups

**Lite (SQLite):**

```bash
docker compose --profile lite exec lite avd data backup --out /data/backup-$(date +%F).db
docker compose --profile lite cp lite:/data/backup-$(date +%F).db .
```

The backup uses SQLite's online backup API, so it is consistent while running. `avd data
restore <file>` restores into a database file that does not exist yet; it never overwrites.

**Full stack (PostgreSQL):**

```bash
docker compose exec -T postgres pg_dump -U avd -Fc avd > avd-$(date +%F).dump
# restore into an empty database:
docker compose exec -T postgres pg_restore -U avd -d avd --clean --if-exists < avd-YYYY-MM-DD.dump
```

Also back up the `media` volume if users submit audio/video. Backups contain authorized
transcripts: store them with the same care as the instance. To share results, use evaluation
exports instead ([`evaluation-export.md`](evaluation-export.md)).

## Upgrades

```bash
git pull
docker compose up -d --build      # migrations run when the API starts
```

* Read the release notes or commit log first. A change in framework, rubric or prompt version
  makes earlier evaluations stale. They are re-evaluated by `avd reevaluate-stale`, which Celery
  beat schedules in the full stack; run it yourself in lite mode. They are never silently mixed.
* Take a backup before upgrading.
* Rolling back the code after a migration needs the backup: `alembic downgrade` exists but has
  not been tested on production data.

## Recovery

| Symptom | Check |
|---|---|
| API container restarting | `docker compose logs api`; in server mode, the startup refusal lists the unsafe settings |
| Evaluations stuck | `avd doctor` (queue, provider); `docker compose logs worker`; for Ollama, `avd local doctor` |
| Ollama down | jobs wait and retry; nothing fails permanently; the API and existing results keep working |
| Lost or corrupt database | restore the latest backup; evaluations can be regenerated from content, but user preferences and feedback cannot |
| Extension "Not connected" | *Preferences → Backend → Re-check connection*: it names host-not-allowed, local-only and TLS problems |

## Data and privacy duties

When you run an instance for others, you hold their preferences, feedback and any transcripts
they submit. Tell them what you keep and for how long ([`privacy.md`](privacy.md) describes what
the software stores). Content rights apply to every instance: see
[`permissions.md`](permissions.md).
