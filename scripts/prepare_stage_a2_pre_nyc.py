"""Correct final-lock seed metadata and freeze the selected pre-NYC state.

This preparation step deliberately has no NYC manifest or dataset path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from datetime import datetime
from pathlib import Path


ROOT = Path("outputs/player1_stage_a2")
MANIFESTS = Path("data/gamus/splits/stage_a2")
FINAL = ROOT / "final"
LOCK_PATH = FINAL / "FINAL_MODEL_LOCK.json"
ARCHIVE_PATH = FINAL / "FINAL_MODEL_LOCK_pre_nyc_consistency_fix.json"
FREEZE_PATH = FINAL / "PRE_NYC_MODEL_FREEZE.json"
CHECKPOINT_PATH = FINAL / "M2_FINAL.pth"
THREE_SEED_REPORT = ROOT / "full_candidates/trial_10_three_seed_stability.json"
TRAIN_MANIFEST = MANIFESTS / "train_core.txt"
HPO_MANIFEST = MANIFESTS / "hpo_dev.txt"
FINAL_DEV_MANIFEST = MANIFESTS / "final_dev_val.txt"

EXPECTED_CHECKPOINT_SHA256 = "6fa4f03dd24726092b75aaf3fa606211c5c66eaaa66ef0dbdbf77eb036bf349f"
EXPECTED_SEED = 1337
EXPECTED_EPOCH = 10
EXPECTED_TRIAL = 10


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for attempt in range(40):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(0.05)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def inspect() -> tuple[dict, dict]:
    required = (
        LOCK_PATH,
        CHECKPOINT_PATH,
        THREE_SEED_REPORT,
        TRAIN_MANIFEST,
        HPO_MANIFEST,
        FINAL_DEV_MANIFEST,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing required pre-NYC artifacts: {missing}")
    actual_checkpoint_hash = sha256(CHECKPOINT_PATH)
    if actual_checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"checkpoint SHA256 mismatch: actual={actual_checkpoint_hash}, expected={EXPECTED_CHECKPOINT_SHA256}"
        )
    lock = load(LOCK_PATH)
    representative = (
        lock.get("three_seed_stability", {})
        .get("outlier_assessment", {})
        .get("selected_representative_seed")
    )
    facts = {
        "model_name": lock.get("model_name"),
        "selected_source_trial": lock.get("selected_source_trial"),
        "selected_seed": lock.get("selected_seed"),
        "selected_epoch": lock.get("selected_epoch"),
        "checkpoint_path": lock.get("checkpoint_path"),
        "checkpoint_SHA256_in_lock": lock.get("checkpoint_SHA256"),
        "checkpoint_SHA256_actual": actual_checkpoint_hash,
        "hyperparameters_seed": lock.get("hyperparameters", {}).get("seed"),
        "three_seed_representative": representative,
        "selected_using_NYC": lock.get("selected_using_NYC"),
        "tuned_using_NYC": lock.get("tuned_using_NYC"),
        "calibrated_using_NYC": lock.get("calibrated_using_NYC"),
        "evaluated_on_NYC": lock.get("evaluated_on_NYC"),
    }
    required_equal = {
        "model_name": "M2-FINAL",
        "selected_source_trial": EXPECTED_TRIAL,
        "selected_seed": EXPECTED_SEED,
        "selected_epoch": EXPECTED_EPOCH,
        "checkpoint_path": str(CHECKPOINT_PATH),
        "checkpoint_SHA256_in_lock": EXPECTED_CHECKPOINT_SHA256,
        "checkpoint_SHA256_actual": EXPECTED_CHECKPOINT_SHA256,
        "three_seed_representative": EXPECTED_SEED,
        "selected_using_NYC": False,
        "tuned_using_NYC": False,
        "calibrated_using_NYC": False,
        "evaluated_on_NYC": False,
    }
    mismatches = {
        key: {"actual": facts.get(key), "expected": expected}
        for key, expected in required_equal.items()
        if facts.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"authoritative final-lock mismatch: {mismatches}")
    return lock, facts


def corrected_lock(lock: dict) -> dict:
    corrected = dict(lock)
    corrected["model_name"] = "M2-FINAL"
    corrected["selected_source_trial"] = EXPECTED_TRIAL
    corrected["selected_seed"] = EXPECTED_SEED
    corrected["selected_epoch"] = EXPECTED_EPOCH
    corrected["checkpoint_path"] = str(CHECKPOINT_PATH)
    corrected["checkpoint"] = str(CHECKPOINT_PATH)
    corrected["checkpoint_SHA256"] = EXPECTED_CHECKPOINT_SHA256
    corrected["checkpoint_sha256"] = EXPECTED_CHECKPOINT_SHA256
    hyperparameters = dict(corrected.get("hyperparameters", {}))
    hyperparameters["seed"] = EXPECTED_SEED
    corrected["hyperparameters"] = hyperparameters

    if "seed_stability_metrics" in corrected:
        if "two_seed_initial_stability_check" in corrected:
            raise RuntimeError("both old and explicitly historical two-seed stability keys exist")
        historical = dict(corrected.pop("seed_stability_metrics"))
        historical["analysis_role"] = "historical initial two-seed check"
        historical["superseded_by_three_seed_analysis"] = True
        corrected["two_seed_initial_stability_check"] = historical
    elif "two_seed_initial_stability_check" not in corrected:
        raise RuntimeError("historical two-seed evidence is missing")

    corrected["selected_using_NYC"] = False
    corrected["tuned_using_NYC"] = False
    corrected["calibrated_using_NYC"] = False
    corrected["evaluated_on_NYC"] = False
    corrected["NYC_statement"] = "NYC has not been used or evaluated."
    corrected["pre_nyc_consistency_audit"] = {
        "timestamp": now_iso(),
        "selected_seed_matches_training_seed_metadata": True,
        "selected_seed": EXPECTED_SEED,
        "training_seed_metadata": EXPECTED_SEED,
        "selected_epoch": EXPECTED_EPOCH,
        "selected_source_trial": EXPECTED_TRIAL,
        "checkpoint_SHA256_verified": EXPECTED_CHECKPOINT_SHA256,
        "three_seed_representative": EXPECTED_SEED,
        "three_seed_stability_authoritative": True,
        "two_seed_check_historical": True,
        "evaluated_on_NYC": False,
    }
    return corrected


def audit_corrected(lock: dict) -> dict:
    historical = lock.get("two_seed_initial_stability_check", {})
    checks = {
        "selected_seed_1337": lock.get("selected_seed") == EXPECTED_SEED,
        "training_seed_metadata_1337": lock.get("hyperparameters", {}).get("seed") == EXPECTED_SEED,
        "selected_epoch_10": lock.get("selected_epoch") == EXPECTED_EPOCH,
        "source_trial_10": lock.get("selected_source_trial") == EXPECTED_TRIAL,
        "checkpoint_path_exact": lock.get("checkpoint_path") == str(CHECKPOINT_PATH),
        "checkpoint_SHA_matches": (
            lock.get("checkpoint_SHA256") == EXPECTED_CHECKPOINT_SHA256 == sha256(CHECKPOINT_PATH)
        ),
        "three_seed_representative_1337": (
            lock.get("three_seed_stability", {})
            .get("outlier_assessment", {})
            .get("selected_representative_seed")
            == EXPECTED_SEED
        ),
        "two_seed_evidence_historical": bool(historical)
        and historical.get("superseded_by_three_seed_analysis") is True,
        "old_ambiguous_two_seed_key_absent": "seed_stability_metrics" not in lock,
        "selected_using_NYC_false": lock.get("selected_using_NYC") is False,
        "tuned_using_NYC_false": lock.get("tuned_using_NYC") is False,
        "calibrated_using_NYC_false": lock.get("calibrated_using_NYC") is False,
        "evaluated_on_NYC_false": lock.get("evaluated_on_NYC") is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"post-fix internal consistency audit failed: {checks}")
    return checks


def create_freeze(lock: dict, checks: dict) -> dict:
    actual_manifest_hashes = {
        "train_manifest_SHA256": sha256(TRAIN_MANIFEST),
        "HPO_dev_manifest_SHA256": sha256(HPO_MANIFEST),
        "FINAL_DEV_VAL_manifest_SHA256": sha256(FINAL_DEV_MANIFEST),
    }
    expected_manifest_hashes = {
        key: lock.get(key) for key in actual_manifest_hashes
    }
    if actual_manifest_hashes != expected_manifest_hashes:
        raise RuntimeError(
            f"pre-NYC manifest hash mismatch: actual={actual_manifest_hashes}, expected={expected_manifest_hashes}"
        )
    freeze = {
        "record_type": "immutable pre-NYC selected-model state",
        "created_at": now_iso(),
        "checkpoint_path": str(CHECKPOINT_PATH),
        "checkpoint_SHA256": sha256(CHECKPOINT_PATH),
        "FINAL_MODEL_LOCK_path": str(LOCK_PATH),
        "FINAL_MODEL_LOCK_SHA256": sha256(LOCK_PATH),
        "selected_seed": lock["selected_seed"],
        "selected_epoch": lock["selected_epoch"],
        "selected_source_trial": lock["selected_source_trial"],
        "train_manifest_path": str(TRAIN_MANIFEST),
        "train_manifest_SHA256": actual_manifest_hashes["train_manifest_SHA256"],
        "HPO_dev_manifest_path": str(HPO_MANIFEST),
        "HPO_dev_manifest_SHA256": actual_manifest_hashes["HPO_dev_manifest_SHA256"],
        "FINAL_DEV_VAL_manifest_path": str(FINAL_DEV_MANIFEST),
        "FINAL_DEV_VAL_manifest_SHA256": actual_manifest_hashes["FINAL_DEV_VAL_manifest_SHA256"],
        "three_seed_stability_report_path": str(THREE_SEED_REPORT),
        "three_seed_stability_report_SHA256": sha256(THREE_SEED_REPORT),
        "lock_consistency_audit": checks,
        "model_selection_frozen": True,
        "evaluated_on_NYC": False,
        "NYC_statement": "NYC has not been used or evaluated at freeze time.",
    }
    dump(FREEZE_PATH, freeze)
    return freeze


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    lock, facts = inspect()
    proposed = corrected_lock(lock)
    proposed_checks = audit_corrected(proposed)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "validated for metadata-only correction",
                    "current": facts,
                    "will_change": {
                        "hyperparameters.seed": [facts["hyperparameters_seed"], EXPECTED_SEED],
                        "seed_stability_metrics": "rename and label historical",
                    },
                    "proposed_audit": proposed_checks,
                    "checkpoint_will_be_modified": False,
                },
                indent=2,
            )
        )
        return

    if ARCHIVE_PATH.exists() or FREEZE_PATH.exists():
        raise RuntimeError("pre-NYC consistency archive or freeze already exists; refusing overwrite")
    shutil.copy2(LOCK_PATH, ARCHIVE_PATH)
    if sha256(ARCHIVE_PATH) != sha256(LOCK_PATH):
        raise RuntimeError("archived lock does not exactly match the pre-fix lock")
    dump(LOCK_PATH, proposed)
    corrected = load(LOCK_PATH)
    checks = audit_corrected(corrected)
    freeze = create_freeze(corrected, checks)
    print(
        json.dumps(
            {
                "status": "PRE_NYC_MODEL_FROZEN",
                "corrected_lock": str(LOCK_PATH),
                "archived_lock": str(ARCHIVE_PATH),
                "freeze": str(FREEZE_PATH),
                "checkpoint_SHA256": freeze["checkpoint_SHA256"],
                "FINAL_MODEL_LOCK_SHA256": freeze["FINAL_MODEL_LOCK_SHA256"],
                "audit": checks,
                "NYC_evaluated": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
