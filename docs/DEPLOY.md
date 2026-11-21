# Running Gauge beyond a desktop

Gauge started as a Windows desktop program. It also runs headless — on a home server, a VPS, or a
container — with an HTTP control API and a small web dashboard, so you can drive it from another machine,
a phone, or a browser. The multi-machine compute pool (`gauge worker`, see COMPUTE.md) already spreads
model calls and training across machines you own; this is the piece that lets the whole engine live
somewhere other than your desk.

## The control server

```
gauge serve                      # http://127.0.0.1:8800, loopback only, prints an API token
gauge serve --host 0.0.0.0       # expose to the network — requires a token, prints a warning
gauge serve --token <your-token> # pin the token instead of generating one
```

It prints the dashboard URL and the API token at start. Open the dashboard, paste the token (it stays in
your browser), and you can read projects, read chapters, watch the newsdesk queue, and trigger the next
chapter.

### API

Every call needs the token in an `X-Gauge-Token` header.

| method | path | what |
| --- | --- | --- |
| GET | `/api/health` | liveness and data dir |
| GET | `/api/projects` | list projects |
| GET | `/api/projects/<name>` | project status |
| GET | `/api/projects/<name>/chapter/<n>` | a chapter's text and meta |
| GET | `/api/newsdesk/queue` | the newsdesk review queue |
| POST | `/api/projects/<name>/write` | write the next chapter |
| POST | `/api/cycle` | run one daemon learning cycle |

```
curl -H "X-Gauge-Token: $TOKEN" http://host:8800/api/projects
curl -X POST -H "X-Gauge-Token: $TOKEN" "http://host:8800/api/projects/My%20Serial/write"
```

## Security

This is a control API: it can write chapters and run cycles. It is built to the same posture as the rest
of Gauge:

- **Loopback by default.** It binds to `127.0.0.1` unless you pass `--host`. On loopback, only programs on
  the same machine can reach it.
- **Token on every call.** No token, no access. A non-loopback bind is *refused* without a token.
- **Guarded.** The same Guard that protects the compute worker applies: per-peer rate limiting and a
  lockout after repeated bad tokens; non-private peers are refused unless you bound to a public address on
  purpose.
- **Never on the open internet.** If you expose it, put it behind a VPN (WireGuard, Tailscale) or a
  reverse proxy that terminates TLS and adds its own auth. The server speaks plain HTTP; it does not
  terminate TLS itself.
- **Lockdown-aware.** The security sentinel's lockdown switches off the risky automation the API can
  trigger, exactly as it does for the daemon.

## Docker

```
export GAUGE_SERVER_TOKEN=$(openssl rand -base64 24)
docker compose up -d                 # publishes to 127.0.0.1:8800 on the host by default
docker compose exec gauge gauge --data-dir /data projects   # run any CLI command inside
```

The image runs as a non-root user and keeps all state on the `gauge-data` volume (projects, models,
config), so it survives restarts and upgrades. The compose file publishes the port to host loopback only;
change the mapping to a public interface **only** behind a VPN or TLS proxy. For a cloud VM, the simplest
safe setup is: bind the container to loopback, run Tailscale on the host, and reach the dashboard over the
tailnet.

## Which machine does what

- **One box, headless:** `gauge serve` on a server; drive it from your laptop or phone over the tailnet.
- **A strong box for the model, a cheap box for the engine:** run the model on the strong machine via
  `gauge worker`, point the engine's `compute.pool_generation` at it, and run `gauge serve` wherever the
  engine lives.
- **Container on a VM:** `docker compose up`, reach it over a VPN. Mount a volume so nothing is lost.
