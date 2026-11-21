"""An opt-in mailing list: people who signed up, imported by the author, emailed with consent.

This is the legitimate, lawful form of email marketing. It will only hold subscribers who have evidence
of consent, it suppresses anyone who unsubscribed and never resurrects them, and every message it builds
carries the sender's identity and a working unsubscribe link — the things CAN-SPAM, GDPR and CASL
require. There is no path here to import a list without consent, and the sender defaults to a dry run, so
nothing goes out until the author supplies their own mail server and asks to send.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from ..db import Database
from ..util import utcnow

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _consent_of(row: dict) -> Optional[str]:
    """Evidence that this person signed up: a consent/signup timestamp, or an explicit consent flag."""
    for key in ("consent_at", "signup_at", "subscribed_at", "opt_in_at", "date"):
        v = row.get(key)
        if v:
            return str(v)
    for key in ("consent", "opt_in", "subscribed"):
        v = row.get(key)
        if str(v).strip().lower() in ("1", "true", "yes", "y"):
            return "confirmed"
    return None


def import_subscribers(db: Database, rows: list[dict], source: str = "import") -> dict[str, int]:
    """Add only rows that carry a valid email AND evidence of consent. Rows without consent are rejected,
    not stored. Returns a summary so the author sees exactly what was and was not accepted."""
    out = {"added": 0, "already": 0, "no_consent": 0, "invalid_email": 0, "kept_unsubscribed": 0}
    for row in rows:
        email = str(row.get("email", "")).strip().lower()
        if not EMAIL_RE.match(email):
            out["invalid_email"] += 1
            continue
        consent = _consent_of(row)
        if not consent:
            out["no_consent"] += 1  # refused: no evidence they signed up
            continue
        name = str(row.get("name", "") or "").strip()
        result = db.add_subscriber(email, name, consent, source)
        out[result if result in out else "added"] = out.get(result if result in out else "added", 0) + 1
    return out


def unsubscribe(db: Database, email: str) -> bool:
    return db.set_subscriber_status(email.strip().lower(), "unsubscribed")


def import_unsubscribes(db: Database, emails: list[str]) -> int:
    n = 0
    for e in emails:
        if db.set_subscriber_status(str(e).strip().lower(), "unsubscribed"):
            n += 1
    return n


@dataclass
class Sender:
    name: str
    email: str
    postal_address: str = ""  # CAN-SPAM requires a physical mailing address in commercial email

    def valid(self) -> tuple[bool, str]:
        if not self.name or not EMAIL_RE.match(self.email or ""):
            return False, "a sender name and a valid sender email are required"
        return True, "ok"


@dataclass
class Newsletter:
    subject: str
    body: str


def build_newsletter(kit: Any, url: str = "", chapter_title: str = "") -> Newsletter:
    title = getattr(kit, "title", "the story")
    subject = f"New from {title}" + (f": {chapter_title}" if chapter_title else "")
    parts = [getattr(kit, "tagline", "").strip(), "", getattr(kit, "blurb_short", "").strip()]
    if chapter_title:
        parts += ["", f"The latest chapter, “{chapter_title}”, is up now."]
    if url:
        parts += ["", f"Read it here: {url}"]
    note = getattr(kit, "author_note", "")
    if note:
        parts += ["", note]
    return Newsletter(subject=subject, body="\n".join(p for p in parts if p is not None).strip())


def render_message(newsletter: Newsletter, to_email: str, sender: Sender, unsubscribe_base: str) -> str:
    """The full email body: the content, the sender's identity, and a one-click unsubscribe link. Both the
    identity and the unsubscribe are mandatory — render_message will not omit them."""
    unsub = unsubscribe_base.rstrip("/") + "/" + to_email if unsubscribe_base else ""
    footer = [f"—\nYou are receiving this because you signed up at {sender.name}."]
    if sender.postal_address:
        footer.append(sender.postal_address)
    footer.append(f"Unsubscribe: {unsub}")
    return newsletter.body + "\n\n" + "\n".join(footer)


class Mailer(Protocol):
    def send(self, to: str, subject: str, body: str, sender: Sender) -> dict[str, Any]:  # pragma: no cover
        ...


@dataclass
class DryRunMailer:
    """The default: records what WOULD be sent and sends nothing."""
    sent: list[dict] = field(default_factory=list)

    def send(self, to: str, subject: str, body: str, sender: Sender) -> dict[str, Any]:
        self.sent.append({"to": to, "subject": subject})
        return {"sent": False, "dry_run": True, "to": to}


@dataclass
class SMTPMailer:
    """Sends through the author's own SMTP server. Not used unless the author constructs it with their
    credentials and passes it explicitly."""
    host: str
    port: int = 587
    username: str = ""
    password: str = ""
    use_tls: bool = True

    def send(self, to: str, subject: str, body: str, sender: Sender) -> dict[str, Any]:  # pragma: no cover - needs a server
        import smtplib
        from email.message import EmailMessage
        msg = EmailMessage()
        msg["From"] = f"{sender.name} <{sender.email}>"
        msg["To"] = to
        msg["Subject"] = subject
        msg.set_content(body)
        with smtplib.SMTP(self.host, self.port) as s:
            if self.use_tls:
                s.starttls()
            if self.username:
                s.login(self.username, self.password)
            s.send_message(msg)
        return {"sent": True, "to": to}


def send_newsletter(db: Database, newsletter: Newsletter, sender: Sender, unsubscribe_base: str,
                    mailer: Optional[Mailer] = None) -> dict[str, Any]:
    """Send only to consented, still-subscribed recipients. Every message carries the sender identity and
    an unsubscribe link. Refuses to run without a valid sender and an unsubscribe base."""
    ok, why = sender.valid()
    if not ok:
        raise PermissionError(why)
    if not unsubscribe_base:
        raise PermissionError("an unsubscribe link base is required in every message; refusing to send without one")
    mailer = mailer or DryRunMailer()
    recipients = db.subscribers(status="subscribed")
    results = {"recipients": len(recipients), "sent": 0, "dry_run": isinstance(mailer, DryRunMailer), "failed": 0, "at": utcnow().isoformat()}
    for sub in recipients:
        body = render_message(newsletter, sub["email"], sender, unsubscribe_base)
        try:
            r = mailer.send(sub["email"], newsletter.subject, body, sender)
            if r.get("sent"):
                results["sent"] += 1
        except Exception:
            results["failed"] += 1
    return results
