"""Promise/payoff heatmap over the whole serial (roadmap 31)."""
from gauge.write.agents.showrunner import Promise, heatmap_text, promise_heatmap


def _promises():
    return [
        Promise(id=1, text="who wrote the name", kind="question", planted=1, due=4, paid=5),
        Promise(id=2, text="the debt comes due", kind="threat", planted=2, due=5, advanced=[3]),
        Promise(id=3, text="will Rin trust Jun", kind="romance", planted=3, due=10),
    ]


def test_heatmap_counts_plant_advance_pay_open_overdue():
    cells = promise_heatmap(_promises(), 6)
    assert len(cells) == 6
    assert cells[0].planted == 1 and cells[0].open_after == 1
    assert cells[2].advanced == 1            # ch3 advanced promise #2
    assert cells[3].overdue_after == 1       # ch4: promise #1 is past its due (4) and unpaid
    assert cells[4].paid == 1                # ch5: promise #1 paid
    assert cells[4].open_after == 2          # #2 and #3 remain
    assert cells[5].overdue_after == 1       # #2 overdue (due 5), unpaid


def test_heatmap_empty_is_safe():
    assert promise_heatmap([], 0) == []
    assert "no promises" in heatmap_text([], 0)


def test_heatmap_text_lists_open_threads_overdue_first():
    out = heatmap_text(_promises(), 6)
    assert "overdue" in out.lower() and "open threads" in out
    # The overdue threat is listed before the not-yet-due romance.
    assert out.index("the debt comes due") < out.index("will Rin trust Jun")
    assert "who wrote the name" not in out.split("open threads")[1]  # paid promise is not an open thread
