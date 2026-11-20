"""One training cycle: features -> engagement model -> exemplars -> style guide -> datasets."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ..config import GaugeConfig
from ..db import Database
from ..features import chapter_features, novel_features
from ..llm.embeddings import Embedder, make_embedder
from ..util import utcnow
from .dataset import export_preference_pairs, export_sft, write_modelfile
from .engagement import TARGETS, EngagementModel, build_training_table
from .exemplars import ExemplarLibrary, make_candidates
from .style import compact_guide_for_prompt, distil_style_guide

log = logging.getLogger(__name__)


class Trainer:
    def __init__(self, cfg: GaugeConfig, db: Database, embedder: Optional[Embedder] = None):
        self.cfg = cfg
        self.db = db
        self.embedder = embedder or make_embedder(cfg.llm)

    @property
    def model_path(self) -> Path:
        return self.cfg.models_dir / f"engagement_{self.cfg.learn.target}.json"

    @property
    def guide_path(self) -> Path:
        return self.cfg.models_dir / "style_guide.md"

    # ---- step 1: features --------------------------------------------------
    def analyze(self, only_new: bool = True) -> dict[str, int]:
        started = utcnow()
        done_chapters = self.db.features("chapter") if only_new else {}
        n_ch = n_nov = 0
        for novel in self.db.novels():
            chapters = self.db.chapters(novel.key)
            if not chapters:
                continue
            feats: list[dict[str, float]] = []
            for ch in chapters:
                key = f"{novel.key}#{ch.index}"
                f = done_chapters.get(key)
                if f is None:
                    f = chapter_features(ch)
                    self.db.save_features(key, "chapter", f)
                    n_ch += 1
                feats.append(f)
            nf = novel_features(novel, feats, [c.published_at for c in chapters])
            self.db.save_features(novel.key, "novel", nf)
            n_nov += 1
        summary = {"chapters_featurized": n_ch, "novels_featurized": n_nov}
        self.db.log_run("analyze", started, summary)
        return summary

    # ---- step 2: model -----------------------------------------------------
    def train_model(self) -> Optional[EngagementModel]:
        novels = self.db.novels()
        nf = self.db.features("novel")
        X, y, names, keys = build_training_table(novels, nf, self.cfg.learn.target)
        if len(y) < 5:
            log.warning("only %d labelled novels; need >= 5 to fit the engagement model", len(y))
            return None
        model = EngagementModel.fit(X, y, names, self.cfg.learn.target, alpha=self.cfg.learn.ridge_alpha)
        model.save(self.model_path)
        return model

    def load_model(self) -> Optional[EngagementModel]:
        return EngagementModel.load(self.model_path)

    # ---- step 3: exemplars --------------------------------------------------
    def build_exemplars(self) -> ExemplarLibrary:
        fn = TARGETS[self.cfg.learn.target]
        cands = []
        chapter_feats = self.db.features("chapter")
        for novel in self.db.novels():
            eng = fn(novel)
            if eng is None:
                continue
            chapters = self.db.chapters(novel.key)
            feats = [chapter_feats.get(f"{novel.key}#{c.index}", {}) for c in chapters]
            cands.extend(make_candidates(novel.key, novel.title, novel.genres + novel.tags, eng, chapters, feats))
        lib = ExemplarLibrary(self.embedder)
        lib.build(cands, keep=self.cfg.learn.exemplar_count)
        lib.save(self.cfg.models_dir)
        return lib

    def load_exemplars(self) -> ExemplarLibrary:
        lib = ExemplarLibrary(self.embedder)
        lib.load(self.cfg.models_dir)
        return lib

    # ---- step 4: style guide + datasets ------------------------------------
    def distil(self, model: EngagementModel) -> str:
        novels = self.db.novels()
        nf = self.db.features("novel")
        fn = TARGETS[self.cfg.learn.target]
        ranked = sorted((n for n in novels if fn(n) is not None), key=lambda n: fn(n), reverse=True)
        top = [n.key for n in ranked[: max(5, len(ranked) // 5)]]
        guide = distil_style_guide(model, nf, top)
        self.guide_path.write_text(guide, encoding="utf-8")
        write_modelfile(self.cfg.llm.model, compact_guide_for_prompt(guide),
                        self.cfg.models_dir / "Modelfile", self.cfg.llm.temperature)
        return guide

    def load_guide(self) -> str:
        return self.guide_path.read_text(encoding="utf-8") if self.guide_path.exists() else ""

    def export_datasets(self) -> dict[str, int]:
        novels = self.db.novels()
        ds = self.cfg.models_dir / "datasets"
        n_pref = export_preference_pairs(self.db, novels, self.cfg.learn.target, ds / "preference.jsonl")
        n_sft = export_sft(self.db, novels, ds / "sft.jsonl")
        return {"preference_pairs": n_pref, "sft_rows": n_sft}

    # ---- the whole cycle ---------------------------------------------------
    def cycle(self) -> dict[str, object]:
        started = utcnow()
        out: dict[str, object] = {}
        out.update(self.analyze())
        if self.db.total_chapters() < self.cfg.learn.min_chapters_for_training:
            out["model"] = f"skipped: {self.db.total_chapters()} chapters < {self.cfg.learn.min_chapters_for_training}"
            self.db.log_run("train", started, out)
            return out
        model = self.train_model()
        if model is None:
            out["model"] = "skipped: too few labelled novels"
        else:
            out["model"] = {"n": model.n_train, "r2_train": round(model.r2_train, 3),
                            "r2_cv": round(model.r2_cv, 3) if model.r2_cv == model.r2_cv else None}
            self.distil(model)
        lib = self.build_exemplars()
        out["exemplars"] = len(lib.items)
        out.update(self.export_datasets())
        self.db.log_run("train", started, out)
        return out
