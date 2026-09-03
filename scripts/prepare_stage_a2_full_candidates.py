"""Freeze the clean HPO evidence and select the three full-training candidates."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import optuna


ROOT = Path("outputs/player1_stage_a2")
HPO_DIR = ROOT / "hpo"
OUT_DIR = ROOT / "full_candidates"
DB_PATH = HPO_DIR / "optuna_stage_a2.db"
STORAGE = f"sqlite:///{DB_PATH.resolve().as_posix()}"
STUDY_NAME = "depthwizard_stage_a2"
MANDATORY = (10, 11)
DIVERSE_PARETO = 26


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def trial_payload(trial: optuna.trial.FrozenTrial, rank: int | None = None) -> dict:
    metrics = trial.user_attrs.get("best_metrics")
    if not isinstance(metrics, dict):
        raise RuntimeError(f"complete trial {trial.number} has no best_metrics")
    payload = {
        "trial_number": trial.number,
        "objective": trial.value,
        "best_epoch": trial.user_attrs.get("best_epoch"),
        "parameters": trial.params,
        "metrics_label": "HPO-DEV PROXY METRICS",
        "best_metrics": metrics,
    }
    if rank is not None:
        payload["objective_rank"] = rank
    return payload


def main() -> None:
    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)
    counts = Counter(trial.state.name for trial in study.trials)
    expected = {"COMPLETE": 20, "PRUNED": 10, "FAIL": 0, "RUNNING": 0}
    actual = {name: counts.get(name, 0) for name in expected}
    if len(study.trials) != 30 or actual != expected:
        raise RuntimeError(f"clean HPO invariant failed: total={len(study.trials)}, states={actual}")
    if study.best_trial.number != 10:
        raise RuntimeError(f"expected clean best trial 10, found {study.best_trial.number}")

    complete = sorted(
        (trial for trial in study.trials if trial.state.name == "COMPLETE"),
        key=lambda trial: float(trial.value),
    )
    ranking = [trial_payload(trial, rank) for rank, trial in enumerate(complete, 1)]
    audit = json.loads((ROOT / "data_audit.json").read_text(encoding="utf-8"))
    intersections = audit["intersections"]
    leakage = {
        "source": "existing pre-NYC Stage A2 data audit; NYC files were not opened by this task",
        "tile_id_overlap_counts": {key: len(value) for key, value in intersections.items()},
        "cross_split_rgb_hash_duplicate_count": len(audit["cross_split_rgb_hash_duplicates"]),
        "nyc_leakage_zero": bool(audit["nyc_leakage_zero"]),
        "manifest_sha256": audit["manifest_sha256"],
    }
    if any(leakage["tile_id_overlap_counts"].values()) or leakage["cross_split_rgb_hash_duplicate_count"]:
        raise RuntimeError(f"existing leakage audit is not clean: {leakage}")

    frozen = {
        "status": "immutable_clean_hpo_evidence",
        "created_at": datetime.now().astimezone().isoformat(),
        "clean_study_name": STUDY_NAME,
        "clean_study_path": str(DB_PATH),
        "clean_study_sha256_at_freeze": sha256(DB_PATH),
        "trial_counts": {"total": len(study.trials), **actual},
        "best_trial": study.best_trial.number,
        "best_objective": study.best_value,
        "top_complete_trials": ranking[:10],
        "complete_trial_ranking_path": str(OUT_DIR / "hpo_complete_rankings.json"),
        "preserved_hpo_artifacts": {
            "best_params": {
                "path": str(HPO_DIR / "best_params.json"),
                "sha256": sha256(HPO_DIR / "best_params.json"),
            },
            "trials_csv": {
                "path": str(HPO_DIR / "trials.csv"),
                "sha256": sha256(HPO_DIR / "trials.csv"),
            },
        },
        "archived_invalid_runs_excluded": True,
        "excluded_archives": [
            str(ROOT / "hpo_invalid_nonfinite_20260902_1123"),
            str(ROOT / "hpo_guard_validation_20260902_1203"),
        ],
        "clean_db_mutated_by_freeze": False,
        "leakage_audit": leakage,
        "nyc_loaded_or_evaluated": False,
    }
    dump(OUT_DIR / "hpo_complete_rankings.json", ranking)
    dump(OUT_DIR / "hpo_frozen_summary.json", frozen)

    by_number = {trial.number: trial for trial in complete}
    selected_numbers = (*MANDATORY, DIVERSE_PARETO)
    if any(number not in by_number for number in selected_numbers):
        raise RuntimeError(f"selected trial is not COMPLETE: {selected_numbers}")
    reasons = {
        10: {
            "reason_selected": "Mandatory clean HPO winner and best balanced objective.",
            "key_strengths": ["best clean HPO objective", "strong correlation", "balanced tall-regime gains"],
            "key_weaknesses": ["persistent negative bias", "47.7% HPO proxy win rate", "tall compression remains"],
        },
        11: {
            "reason_selected": "Mandatory Pareto candidate: slightly better RMSE and R2 than Trial 10 despite worse MAE.",
            "key_strengths": ["better RMSE than Trial 10", "better R2 than Trial 10", "better 20-50m MAE"],
            "key_weaknesses": ["worse MAE and building MAE than Trial 10", "lower HPO proxy win rate"],
        },
        26: {
            "reason_selected": "Diverse Pareto candidate with the best HPO MAE, RMSE, R2, Pearson, and 20-50m MAE; uses Cosine plus an unconstrained output instead of duplicating Trials 10/11.",
            "key_strengths": ["best HPO MAE/RMSE/R2", "best 20-50m MAE", "hyperparameter and output-head diversity"],
            "key_weaknesses": ["worse balanced objective than Trial 10", "lower Spearman", "negative bias and tall compression remain"],
        },
    }
    candidates = []
    for candidate_index, number in enumerate(selected_numbers, 1):
        row = trial_payload(by_number[number])
        row.update({"candidate_index": candidate_index, **reasons[number]})
        candidates.append(row)
    selection = {
        "status": "frozen_exactly_three_candidates",
        "created_at": datetime.now().astimezone().isoformat(),
        "candidate_count": 3,
        "selection_scope": "clean COMPLETE HPO trials only",
        "candidate_3_method": "Pareto evidence plus hyperparameter diversity; objective rank alone was not used",
        "candidates": candidates,
        "hpo_metrics_label": "HPO-DEV PROXY METRICS",
        "hpo_proxy_wins_are_final_tile_wins": False,
        "nyc_used": False,
    }
    dump(OUT_DIR / "candidate_selection.json", selection)
    print(json.dumps({"frozen": str(OUT_DIR / "hpo_frozen_summary.json"), "selected_trials": selected_numbers}, indent=2))


if __name__ == "__main__":
    main()
