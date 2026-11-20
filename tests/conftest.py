import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gauge.config import GaugeConfig  # noqa: E402
from gauge.db import Database  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def cfg(tmp_path):
    c = GaugeConfig(data_dir=str(tmp_path / "data"))
    c.llm.provider = "fake"
    c.learn.min_chapters_for_training = 5
    c.ensure_dirs()
    return c


@pytest.fixture
def db(cfg):
    d = Database(cfg.db_path)
    yield d
    d.close()


def _chapter_text(hooky: bool, n_pars: int = 30) -> str:
    """Synthetic prose generator: 'hooky' chapters open on dialogue/action and end unresolved."""
    opening = ("\"Run,\" she said.\n\nThe blade was already in the door before I understood what she meant.\n\n"
               if hooky else
               "It was a pleasant afternoon in the village, and the weather had been mild for the season.\n\n"
               "The fields were green and the farmers were content with the slow progress of the harvest.\n\n")
    middle = ("The corridor stank of smoke. Someone screamed below us. I counted the steps, lost count, "
              "counted again.\n\n\"Left,\" she said. \"No. Right.\"\n\nWe went right.\n\n") if hooky else (
              "We discussed the matter at considerable length over tea and arrived at no particular "
              "conclusion, which was agreeable to everyone present and in keeping with local custom.\n\n")
    ending = ("The door at the end of the hall opened on its own.\n\nMy name was written on the wall inside, "
              "in my own handwriting.\n\nAnd below it, a date: tomorrow." if hooky else
              "Afterwards we all went home and slept well, and nothing of note happened for some time. "
              "The matter was fully resolved and everyone was satisfied with how it had turned out.")
    body = opening + middle * max(1, n_pars // 3) + ending
    return body


@pytest.fixture
def chapter_text():
    return _chapter_text
