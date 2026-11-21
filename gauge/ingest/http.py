"""Polite HTTP fetcher: one user agent, rate limiting, robots.txt, retries."""
from __future__ import annotations

import logging
import time
import urllib.robotparser as robotparser
from typing import Optional
from urllib.parse import urlparse

import requests

from ..config import IngestConfig

log = logging.getLogger(__name__)


class Fetcher:
    def __init__(self, cfg: IngestConfig, session: Optional[requests.Session] = None):
        self.cfg = cfg
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": cfg.user_agent, "Accept-Language": "en"})
        self._last_request: dict[str, float] = {}
        self._robots: dict[str, Optional[robotparser.RobotFileParser]] = {}

    # ---- robots ----------------------------------------------------------------
    def _robots_for(self, url: str) -> Optional[robotparser.RobotFileParser]:
        host = urlparse(url).netloc
        if host in self._robots:
            return self._robots[host]
        rp: Optional[robotparser.RobotFileParser] = robotparser.RobotFileParser()
        try:
            r = self.session.get(f"{urlparse(url).scheme}://{host}/robots.txt", timeout=15)
            if r.ok:
                rp.parse(r.text.splitlines())
            else:
                rp = None
        except requests.RequestException:
            rp = None
        self._robots[host] = rp
        return rp

    def allowed(self, url: str) -> bool:
        if not self.cfg.respect_robots_txt:
            return True
        rp = self._robots_for(url)
        if rp is None:
            return True
        return rp.can_fetch(self.cfg.user_agent, url)

    # ---- fetch -----------------------------------------------------------------
    def _throttle(self, url: str) -> None:
        host = urlparse(url).netloc
        last = self._last_request.get(host, 0.0)
        wait = self.cfg.request_delay_seconds - (time.monotonic() - last)
        if wait > 0:
            time.sleep(wait)
        self._last_request[host] = time.monotonic()

    def get(self, url: str, retries: int = 4, **kwargs) -> Optional[str]:
        if not self.allowed(url):
            log.warning("robots.txt disallows %s; skipping", url)
            return None
        for attempt in range(retries):
            self._throttle(url)
            try:
                r = self.session.get(url, timeout=30, **kwargs)
                if r.status_code == 429 or r.status_code >= 500:
                    # Honour Retry-After when the server sends one (api.wikimedia.org does), else back off.
                    wait = self.cfg.request_delay_seconds * (2 ** attempt) * 2
                    ra = r.headers.get("Retry-After")
                    if ra and ra.isdigit():
                        wait = max(wait, min(30.0, float(ra)))
                    log.info("GET %s -> %s; waiting %.1fs", url, r.status_code, wait)
                    time.sleep(wait)
                    continue
                if not r.ok:
                    log.warning("GET %s -> %s", url, r.status_code)
                    return None
                return r.text
            except requests.RequestException as e:
                log.warning("GET %s failed (%s), attempt %d", url, e, attempt + 1)
                time.sleep(self.cfg.request_delay_seconds * (attempt + 1))
        return None

    def post(self, url: str, data: dict[str, str], retries: int = 2) -> Optional[str]:
        if not self.allowed(url):
            return None
        for _attempt in range(retries):
            self._throttle(url)
            try:
                r = self.session.post(url, data=data, timeout=30)
                if r.ok:
                    return r.text
            except requests.RequestException as e:
                log.warning("POST %s failed (%s)", url, e)
            time.sleep(self.cfg.request_delay_seconds)
        return None
