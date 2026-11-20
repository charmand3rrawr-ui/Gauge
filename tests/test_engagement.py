import numpy as np

from gauge.learn.engagement import EngagementModel, build_training_table
from gauge.models import NovelRecord


def _synthetic(n=60, seed=1):
    rng = np.random.default_rng(seed)
    novels, feats = [], {}
    for i in range(n):
        hook = rng.uniform(0, 1)
        cliff = rng.uniform(0, 1)
        words = rng.uniform(1000, 4000)
        noise = rng.normal(0, 0.2)
        followers = int(np.exp(4 + 2.0 * cliff + 1.0 * hook + noise))
        nv = NovelRecord(source="t", source_id=str(i), title=f"N{i}", followers=followers)
        novels.append(nv)
        feats[nv.key] = {"mean_hook": hook, "mean_cliff": cliff, "mean_words": words, "junk": rng.uniform()}
    return novels, feats


def test_fit_recovers_signal_and_persists(tmp_path):
    novels, feats = _synthetic()
    X, y, names, keys = build_training_table(novels, feats, "log_followers")
    assert X.shape == (60, 4) and len(keys) == 60
    m = EngagementModel.fit(X, y, names, "log_followers", alpha=0.5, try_sklearn=False)
    imps = dict(m.importances())
    assert imps["mean_cliff"] > imps["mean_hook"] > abs(imps["junk"])
    assert m.r2_train > 0.8
    assert m.r2_cv > 0.7
    p = tmp_path / "m.json"
    m.save(p)
    m2 = EngagementModel.load(p)
    sample = {"mean_hook": 0.9, "mean_cliff": 0.9, "mean_words": 2000, "junk": 0.5}
    assert abs(m.predict(sample) - m2.predict(sample)) < 1e-9
    assert 0.5 < m2.score01(sample) <= 1.0
    low = {"mean_hook": 0.1, "mean_cliff": 0.1, "mean_words": 2000, "junk": 0.5}
    assert m2.score01(low) < m2.score01(sample)


def test_contributions_sorted_by_magnitude():
    novels, feats = _synthetic(40, seed=3)
    X, y, names, _ = build_training_table(novels, feats, "log_followers")
    m = EngagementModel.fit(X, y, names, "log_followers", try_sklearn=False)
    c = m.contributions({"mean_hook": 1.0, "mean_cliff": 0.0, "mean_words": 3000, "junk": 0.0})
    mags = [abs(v) for _, v in c]
    assert mags == sorted(mags, reverse=True)


def test_skips_unlabelled():
    novels, feats = _synthetic(10)
    novels[0].followers = None
    X, y, _, keys = build_training_table(novels, feats, "log_followers")
    assert len(y) == 9 and novels[0].key not in keys
