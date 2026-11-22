"""Operations: back up and restore the data directory with integrity, and audit what Gauge can do.

These exist because Gauge now runs as a deployed platform, not just a desktop toy: a server loses its disk,
a config schema changes between versions, and an operator needs to know in plain language exactly which
automations are on and what each can reach.
"""
from .backup import make_backup, restore_backup, verify_backup
from .budget import BudgetStatus, ErrorBudget, budget_for
from .dashboard import dashboard_html
from .logs import event, read_log, redact, remove_structured_log, setup_structured_log
from .report import audit_report, render_report

__all__ = ["make_backup", "verify_backup", "restore_backup", "audit_report", "render_report", "dashboard_html",
           "ErrorBudget", "BudgetStatus", "budget_for", "event", "read_log", "redact", "setup_structured_log",
           "remove_structured_log"]
