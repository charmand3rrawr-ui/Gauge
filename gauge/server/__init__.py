"""Run Gauge headless and drive it over HTTP: a token-authenticated control API and a small dashboard.

Secure by default: it binds to loopback, every API call needs a token, and the same Guard that protects
the compute worker (rate limiting, failed-auth lockout, private-peer-only unless you opt in) protects
this. Exposing it to a network is an explicit, warned choice. See docs/DEPLOY.md.
"""
from .app import ServerState, build_server, serve

__all__ = ["ServerState", "build_server", "serve"]
