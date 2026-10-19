# Security policy

## Reporting a vulnerability

Please **do not open a public issue** for a vulnerability. Use GitHub's private vulnerability
reporting for this repository (Security → Report a vulnerability) if it is enabled. If it is not,
open an issue that says only "security contact requested", and the owner will reach out.

Include the affected version or commit, how to reproduce it, and its impact. You will get an
acknowledgement when the report is read. This is a volunteer project with no response-time
guarantee and no bug bounty.

## Supported versions

Only the latest commit on `main` is supported. There are no release branches yet.

## Scope and deployment boundaries

The threat model and the safe deployment boundaries are in [`docs/security.md`](docs/security.md):

* local mode is single-user, loopback only;
* server mode needs an admin token, allowed hosts and a registration policy;
* reverse proxies terminate TLS;
* model servers, databases and Redis are never exposed publicly.

Reports about a configuration that those documents say is unsupported (for example, a local
instance port-forwarded to the internet) are still welcome as hardening suggestions.
