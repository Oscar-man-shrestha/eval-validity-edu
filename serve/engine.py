"""Load (or fit) the Review–Advance ranker for the demo API."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from junyi_pipeline import DATA, OUT, SEED, history_state, load_graph, load_names, load_sequences
from run_all_phases import (
    build_heads,
    mode_features,
    phase3_memory,
    train_models,
)

BUNDLE = OUT / "serve_models.joblib"

SCENARIOS = [
    {
        "id": "sticky_review",
        "title": "Stuck student",
        "blurb": "Just failed factoring — expect “practice again,” not a brand-new topic.",
        "seed": [
            ("one step equations", 1),
            ("one step inequalities", 1),
            ("solving quadratics by factoring", 0),
        ],
    },
    {
        "id": "ready_to_advance",
        "title": "Student doing well",
        "blurb": "Several correct answers — more room to suggest something new.",
        "seed": [
            ("addition 1", 1),
            ("addition 2", 1),
            ("subtraction 1", 1),
            ("subtraction 2", 1),
            ("multiplication 1", 1),
        ],
    },
    {
        "id": "quadratic_path",
        "title": "Explore the quadratic map",
        "blurb": "See prerequisite parents and children around parabolas.",
        "seed": [
            ("one step equations", 1),
            ("one step inequalities", 1),
            ("solving quadratics by factoring", 1),
            ("graphing parabolas 1", 0),
        ],
    },
]


class Recommender:
    def __init__(self) -> None:
        self.names = load_names()
        self.id_of = {name.lower(): i for i, name in self.names.items()}
        self.graph, self.similar = load_graph(len(self.names))
        self.embeddings = np.load(OUT / "concept_embeddings.npy")
        self.parent_lists = [list(self.graph.predecessors(c)) for c in range(len(self.names))]
        bundle = self._load_or_fit()
        self.logreg = bundle["logreg"]
        self.mode_clf = bundle["mode_clf"]
        self.review_clf = bundle["review_clf"]
        self.advance_clf = bundle["advance_clf"]
        self.popularity = np.asarray(bundle["popularity"])
        self.feat_fn = build_heads(
            self.embeddings, self.graph, self.similar, self.popularity, self.parent_lists, self.logreg
        )
        self.metrics = self._metrics()

    def _load_or_fit(self) -> dict:
        if BUNDLE.exists():
            return joblib.load(BUNDLE)
        rng = np.random.default_rng(SEED)
        train_seq = load_sequences(DATA / "train.json")
        popularity = np.zeros(len(self.names))
        for seq in train_seq:
            for concept, _ in seq:
                popularity[concept] += 1
        logreg, _, _ = phase3_memory(train_seq, rng)
        mode_clf, review_clf, advance_clf, _, _ = train_models(
            train_seq, self.embeddings, self.graph, self.similar, popularity, self.parent_lists, logreg, rng
        )
        bundle = {
            "logreg": logreg,
            "mode_clf": mode_clf,
            "review_clf": review_clf,
            "advance_clf": advance_clf,
            "popularity": popularity,
        }
        joblib.dump(bundle, BUNDLE)
        return bundle

    def _metrics(self) -> dict:
        out: dict = {}
        for name in (
            "phase1_prepare.json",
            "phase2_embeddings.json",
            "phase3_memory.json",
            "phase4_graph.json",
            "phase5_ranker.json",
            "phase6_eval.json",
        ):
            path = OUT / "phases" / name
            if path.exists():
                out[name.replace(".json", "")] = json.loads(path.read_text())
        summary = OUT / "all_phases_results.txt"
        out["headline"] = summary.read_text() if summary.exists() else ""
        audit_path = OUT / "phases" / "research_integrity_audit.json"
        if audit_path.exists():
            out["research_integrity_audit"] = json.loads(audit_path.read_text())
        out["cards"] = self._metric_cards(out)
        return out

    def _metric_cards(self, raw: dict) -> list[dict]:
        audit = raw.get("research_integrity_audit") or {}
        findings = audit.get("validated_findings") or {}
        if findings:
            rec = findings.get("ragr_vs_recent5") or {}
            ready = findings.get("readiness_correctness") or {}
            forget = findings.get("forgetting_gap_ge_1_day") or {}
            repetition = (findings.get("cross_dataset_concept_repetition") or {}).get("rows") or []
            junyi = next((r for r in repetition if r.get("dataset_unit") == "Junyi topic"), {})
            return [
                {
                    "label": "Junyi topic continuation",
                    "value": f"{(junyi.get('share_next_equals_last') or {}).get('share', 0):.0%}",
                    "hint": "This is a dataset property, not evidence that a teaching policy works.",
                },
                {
                    "label": "Prototype vs Recent-5 (R@5)",
                    "value": f"{rec.get('paired_delta_r5', {}).get('mean', 0):+.4f}",
                    "hint": f"Practical bar {rec.get('practical_threshold_r5', 0):.2f}; status: {rec.get('status', 'unavailable')}.",
                },
                {
                    "label": "DAG readiness ΔAUC",
                    "value": f"{ready.get('primary', {}).get('mean', 0):+.4f}",
                    "hint": f"Status: {ready.get('status', 'unavailable')}; this is not a causal learning effect.",
                },
                {
                    "label": "Forgetting with day gaps",
                    "value": f"{forget.get('delta_log_gap_minus_success', {}).get('mean', 0):+.4f}",
                    "hint": f"Status: {forget.get('status', 'unavailable')} on gap≥1-day returns.",
                },
            ]
        p2 = raw.get("phase2_embeddings") or {}
        p3 = raw.get("phase3_memory") or {}
        p4 = raw.get("phase4_graph") or {}
        p6 = (raw.get("phase6_eval") or {}).get("ranking") or {}
        te = p3.get("evaluate_on_test_json") or {}
        td = p3.get("timed_learner_split") or {}
        mix = p4.get("next_item_mix_train_sample") or {}
        cards = [
            {
                "label": "Next click is review",
                "value": f"{mix.get('review_frac', 0):.0%}",
                "hint": "Why overall Recall@K mostly measures restudy",
            },
            {
                "label": "Next == last item",
                "value": f"{p6.get('share_next_equals_last', 0):.0%}",
                "hint": "Sticky-click ceiling; last-item baseline is hard",
            },
            {
                "label": "Memory AUC (test.json)",
                "value": f"{(te.get('logistic_step_dt') or {}).get('auc', 0):.3f}",
                "hint": f"logistic > success {(te.get('success_rate') or {}).get('auc', 0):.3f} > HLR {(te.get('hlr_step') or {}).get('auc', 0):.3f}",
            },
            {
                "label": "RAGR Recall@5",
                "value": f"{(p6.get('ragr_all') or {}).get('mean', 0):.3f}",
                "hint": f"vs force-review {(p6.get('force_review_all') or {}).get('mean', 0):.3f}; CI on Δ above 0",
            },
            {
                "label": "Embeddings vs random",
                "value": f"{p2.get('pair_auc_vs_random', 0):.3f}",
                "hint": f"same-topic control {p2.get('pair_auc_vs_same_topic', 0):.3f} — report both",
            },
        ]
        if td:
            delta = td.get("paired_logistic_day_minus_success") or {}
            cards.append(
                {
                    "label": "Timed forgetting ΔAUC",
                    "value": f"{delta.get('mean_delta_auc', 0):+.4f}",
                    "hint": "day Δt − success; forgetting_helps=false",
                }
            )
        if "ragr_nonlast" in p6:
            cards.append(
                {
                    "label": "Non-last R@5",
                    "value": f"{p6['ragr_nonlast'].get('mean', 0):.3f}",
                    "hint": "Harder slice where next ≠ last click",
                }
            )
        return cards

    def search(self, query: str, limit: int = 12) -> list[dict]:
        q = query.strip().lower()
        hits = []
        for idx, name in self.names.items():
            if not q or q in name.lower():
                hits.append({"id": idx, "name": name})
            if len(hits) >= limit:
                break
        return hits

    def _resolve_name(self, name: str) -> tuple[int, str] | None:
        key = name.strip().lower()
        if key in self.id_of:
            idx = self.id_of[key]
            return idx, self.names[idx]
        # exact token match among names (avoid prefix collisions)
        for idx, nm in self.names.items():
            if nm.lower() == key:
                return idx, nm
        hits = self.search(key, limit=5)
        for h in hits:
            if h["name"].lower() == key:
                return h["id"], h["name"]
        return (hits[0]["id"], hits[0]["name"]) if hits else None

    def scenarios(self) -> list[dict]:
        out = []
        for sc in SCENARIOS:
            hist = []
            for name, correct in sc["seed"]:
                resolved = self._resolve_name(name)
                if resolved:
                    idx, nm = resolved
                    hist.append({"id": idx, "name": nm, "correct": correct})
            out.append({**{k: sc[k] for k in ("id", "title", "blurb")}, "history": hist})
        return out

    def _parse_history(self, events: list[dict]) -> list[tuple[int, int]]:
        seq = []
        for event in events:
            if event.get("id") is not None:
                idx = int(event["id"])
            else:
                key = str(event.get("name", "")).lower()
                if key not in self.id_of:
                    raise ValueError(f"unknown concept {event}")
                idx = self.id_of[key]
            seq.append((idx, int(event.get("correct", 1))))
        return seq

    def recommend(self, events: list[dict], k: int = 8) -> dict:
        seq = self._parse_history(events)
        if len(seq) < 2:
            raise ValueError("need at least two practice events")
        cut = len(seq)
        stats, last_step, recent = history_state(seq, cut)
        query = self.embeddings[recent[-5:]].mean(axis=0)
        query = query / max(np.linalg.norm(query), 1e-8)
        mastery = {c: (s / a if a else 0.0) for c, (a, s) in stats.items()}
        last_concept = recent[-1]
        last_correct = int(seq[-1][1])
        review, advance, seen, p_mem = self.feat_fn(cut, last_concept, stats, last_step, query, mastery)
        gate_x = mode_features(seq, cut, last_step, stats, self.graph)
        p_rev = float(self.mode_clf.predict_proba(gate_x[None, :])[0, 1])
        s_rev = self.review_clf.decision_function(review)
        s_adv = self.advance_clf.decision_function(advance)
        s_rev = np.where(seen, s_rev, s_rev.min() - 1.0)
        mixed = p_rev * s_rev + (1.0 - p_rev) * s_adv
        order = np.argsort(-mixed)[:k]

        why_mode = []
        if last_correct == 0:
            why_mode.append("Last attempt was wrong → restudy pressure rises.")
        else:
            why_mode.append("Last attempt was correct → more room to unlock.")
        if p_rev >= 0.7:
            why_mode.append("High P(review) matches Junyi’s ~90% restudy base rate.")
        elif p_rev < 0.5:
            why_mode.append("Gate leans advance — rare on this log; inspect the amber cards.")
        else:
            why_mode.append("Gate is mixed — both heads can appear in the top-k.")

        recs = []
        for rank, idx in enumerate(order, start=1):
            kind = "review" if seen[idx] else "advance"
            parents = [self.names[p] for p in self.graph.predecessors(idx)]
            unlock = float(
                np.mean([mastery.get(p, 0.0) for p in self.graph.predecessors(idx)])
                if list(self.graph.predecessors(idx))
                else 1.0
            )
            reasons = []
            if kind == "review":
                if idx == last_concept:
                    reasons.append("Same as last item (sticky restudy is common here).")
                if p_mem[idx] < 0.55:
                    reasons.append(f"Low predicted recall ({p_mem[idx]:.2f}) → desirable difficulty.")
                elif p_mem[idx] > 0.85:
                    reasons.append(f"High predicted recall ({p_mem[idx]:.2f}) — already strong.")
                reasons.append("Scored by the review head (memory + ZPD + semantics).")
            else:
                if list(self.graph.predecessors(idx)):
                    reasons.append(f"Soft unlock from parents ≈ {unlock:.2f}.")
                else:
                    reasons.append("DAG root / no parents.")
                if idx in self.graph.successors(last_concept):
                    reasons.append("Direct DAG child of the last exercise.")
                reasons.append("Scored by the advance head (DAG + similarity + popularity).")
            recs.append(
                {
                    "id": int(idx),
                    "rank": rank,
                    "name": self.names[int(idx)],
                    "kind": kind,
                    "score": float(mixed[idx]),
                    "s_review": float(s_rev[idx]),
                    "s_advance": float(s_adv[idx]),
                    "contrib_review": float(p_rev * s_rev[idx]),
                    "contrib_advance": float((1.0 - p_rev) * s_adv[idx]),
                    "p_recall": float(p_mem[idx]),
                    "parents": parents,
                    "unlock": unlock,
                    "reasons": reasons,
                }
            )

        focus = {last_concept}
        focus.update(self.graph.predecessors(last_concept))
        focus.update(self.graph.successors(last_concept))
        for p in list(self.graph.predecessors(last_concept)):
            focus.update(self.graph.predecessors(p))
        for c in list(self.graph.successors(last_concept)):
            focus.update(self.graph.successors(c))
        focus.update(int(i) for i in order[:5])
        # Keep graph readable
        if len(focus) > 22:
            keep = {last_concept}
            keep.update(self.graph.predecessors(last_concept))
            keep.update(self.graph.successors(last_concept))
            keep.update(int(i) for i in order[:5])
            focus = keep
        nodes = [
            {
                "id": i,
                "name": self.names[i],
                "current": i == last_concept,
                "seen": bool(seen[i]),
                "recommended": i in set(order[:5]),
            }
            for i in focus
        ]
        edges = [{"source": u, "target": v} for u, v in self.graph.subgraph(focus).edges()]

        n_rev = sum(1 for r in recs if r["kind"] == "review")
        return {
            "p_review": p_rev,
            "p_advance": 1.0 - p_rev,
            "mode": "review" if p_rev >= 0.5 else "advance",
            "last_concept": {"id": last_concept, "name": self.names[last_concept], "correct": last_correct},
            "history_len": len(seq),
            "seen": len(last_step),
            "top_mix": {"review": n_rev, "advance": len(recs) - n_rev},
            "formula": "score(c) = P(review)·s_rev(c) + (1−P(review))·s_adv(c)",
            "why_mode": why_mode,
            "gate_readout": {
                "last_correct": last_correct,
                "seen_items": len(last_step),
                "history_steps": cut,
                "has_dag_children": bool(list(self.graph.successors(last_concept))),
            },
            "recommendations": recs,
            "subgraph": {"nodes": nodes, "edges": edges},
        }


_engine: Recommender | None = None


def get_engine() -> Recommender:
    global _engine
    if _engine is None:
        _engine = Recommender()
    return _engine


def reload_engine() -> Recommender:
    global _engine
    _engine = Recommender()
    return _engine
