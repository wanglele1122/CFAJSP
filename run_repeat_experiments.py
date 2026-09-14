"""Batch runner for repeated random experiments on a single instance.

It writes one summary JSON and one convergence JSONL per run, plus a batch-level
manifest and aggregate summary for later analysis.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from src.io.data_loader import DataLoader
from src.io.run_report import build_run_report, export_training_records_from_trace
from src.models.base_classes import SchedulingProblem
from src.scheduling.abc_dispatcher import ABCDispatcher
from src.scheduling.neighborhood import EmployedConfig, ScoutConfig
from src.scheduling.pbiga_adapt import PBIGAAdaptConfig, PBIGAAdaptDispatcher
from src.scheduling.decoder import plot_gantt
from src.scheduling.progressive_insertion import find_top_risk_path, _compute_job_urgency
from run_hqpso_vns_demo import run_hqpso_vns
from src.scheduling.single_obj_baselines import (
    SingleObjectiveABCBaseline,
    SingleObjectiveSABaseline,
    SingleObjectiveASAPaperBaseline,
    SingleObjectivePSOBaseline,
)
from src.scheduling.ta_ma import TAMAConfig, TAMADispatcher
from src.tool.asap_scheduler import asap_schedule
from src.tool.buffer_budget import compute_order_budgets


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


DEFAULT_REPEAT_CONFIGS: dict[str, str] = {
    "abc": "/data/Wanglele/ABC_schedule/data/inputs/configures/repeat/abc_v0.json",
    "abc_baseline": "/data/Wanglele/ABC_schedule/data/inputs/configures/repeat/abc_baseline.json",
    "pso_baseline": "/data/Wanglele/ABC_schedule/data/inputs/configures/repeat/pso.json",
    "sa_baseline": "/data/Wanglele/ABC_schedule/data/inputs/configures/repeat/sa.json",
    "asa_paper_baseline": "/data/Wanglele/ABC_schedule/data/inputs/configures/repeat/asa.json",
    "hqpso_vns": "/data/Wanglele/ABC_schedule/data/inputs/configures/repeat/hqpso.json",
    "pbiga_adapt": "/data/Wanglele/ABC_schedule/data/inputs/configures/repeat/pbiga_adapt.json",
    "tama": "/data/Wanglele/ABC_schedule/data/inputs/configures/repeat/tama.json",

}


def extract_suffix_number(file_path: str) -> int | None:
    name = Path(file_path).name
    import re

    match = re.search(r"(\d+)(?=\.json$)", name)
    return int(match.group(1)) if match else None


def resolve_config_path(algorithm: str, config_path: str | None) -> Path:
    if config_path:
        return Path(config_path)

    algo_name = str(algorithm).strip().lower()
    default_path = DEFAULT_REPEAT_CONFIGS.get(algo_name)
    if default_path is None:
        raise ValueError(f"unsupported algorithm for config resolution: {algorithm}")
    return Path(default_path)


def make_config_tag(file_path: str) -> str:
    config_number = extract_suffix_number(file_path)
    if config_number is not None:
        return f"C{config_number}"
    return f"C{Path(file_path).stem}"


def make_emp_cfg(config: dict[str, Any]) -> EmployedConfig:
    return EmployedConfig(
        seq_ops_per_source=config.get("em_seq_ops_per_source"),
        seq_window=config.get("em_seq_window"),
        seq_use_swap_prob=config.get("em_seq_use_swap_prob"),
        risk_bias_ratio=config.get("risk_bias_ratio", 0.7),
        machine_ops_per_source=config.get("em_machine_ops_per_source"),
        buffer_ops_per_source=config.get("em_buffer_ops_per_source"),
        buffer_k_ops=config.get("em_buffer_k_ops"),
        buffer_delta_ratio=config.get("em_buffer_delta_ratio"),
        buffer_shift_prob=config.get("buffer_shift_prob", 0.25),
        integer_buffer=config.get("integer_buffer", False),
        buffer_unit=config.get("buffer_unit", 0.5),
        use_risk_elite=config.get("use_risk_elite", True),
        elite_mode=config.get("elite_mode", "legacy"),
        elite_pool_ratio=config.get("elite_pool_ratio", 0.2),
        elite_candidate_ratio=config.get("elite_candidate_ratio", 0.2),
        elite_machine_frac=config.get("elite_machine_frac", 0.15),
        elite_seq_frac=config.get("elite_seq_frac", 0.15),
        elite_seq_span=config.get("elite_seq_span", 5),
        elite_buffer_blend=config.get("elite_buffer_blend", 0.3),
        elite_imitation_prob=config.get("elite_imitation_prob", 1.0),
        elite_risk_mix_ratio=config.get("elite_risk_mix_ratio", 0.8),
        elite_machine_copy_prob=config.get("elite_machine_copy_prob", 0.7),
        elite_tardy_tail_allow_increase=config.get("elite_tardy_tail_allow_increase", False),
        elite_tardy_tail_increase_cap_ratio=config.get("elite_tardy_tail_increase_cap_ratio", 0.02),
        post_elite_mutation_prob=config.get("post_elite_mutation_prob", 0.8),
        post_elite_seq_prob=config.get("post_elite_seq_prob", 0.5),
        post_elite_machine_prob=config.get("post_elite_machine_prob", 0.3),
        post_elite_buffer_prob=config.get("post_elite_buffer_prob", 0.2),
    )


def make_onl_cfg(config: dict[str, Any]) -> EmployedConfig:
    return EmployedConfig(
        seq_ops_per_source=config.get("on_seq_ops_per_source"),
        seq_window=config.get("on_seq_window"),
        seq_use_swap_prob=config.get("on_seq_use_swap_prob"),
        risk_bias_ratio=config.get("onl_risk_bias_ratio", config.get("risk_bias_ratio", 0.85)),
        machine_ops_per_source=config.get("on_machine_ops_per_source"),
        buffer_ops_per_source=config.get("on_buffer_ops_per_source"),
        buffer_k_ops=config.get("on_buffer_k_ops"),
        buffer_delta_ratio=config.get("on_buffer_delta_ratio"),
        buffer_shift_prob=config.get("on_buffer_shift_prob", config.get("buffer_shift_prob", 0.25)),
        integer_buffer=config.get("integer_buffer", False),
        buffer_unit=config.get("buffer_unit", 0.5),
        use_risk_elite=config.get("use_risk_elite_onl", False),
        elite_mode=config.get("elite_mode_onl", "off"),
        elite_pool_ratio=config.get("elite_pool_ratio_onl", 0.0),
        elite_imitation_prob=config.get("elite_imitation_prob", 0.0),
        onl_p_tardy_promote=config.get("onl_p_tardy_promote", 0.25),
        onl_p_critical_insert=config.get("onl_p_critical_insert", 0.25),
        onl_p_critical_reorder=config.get("onl_p_critical_reorder", 0.15),
        onl_p_bottleneck_machine=config.get("onl_p_bottleneck_machine", 0.15),
        onl_p_job_buffer=config.get("onl_p_job_buffer", 0.15),
        onl_p_job_buffer_shift=config.get("onl_p_job_buffer_shift", 0.10),
        onl_p_elite_local=config.get("onl_p_elite_local", 0.05),
    )


def make_sco_cfg(config: dict[str, Any]) -> ScoutConfig:
    p_window_swap = config.get("p_window_swap", config.get("p_block_shift"))
    p_window_insert = config.get("p_window_insert", config.get("p_block_extract"))
    p_window_reorder = config.get("p_window_reorder", config.get("p_block_reorder"))
    p_window_buffer = config.get("p_window_buffer", config.get("p_group_buffer"))
    p_window_machine = config.get("p_window_machine", config.get("p_path_machine"))
    return ScoutConfig(
        path_mode=str(config.get("scout_path_mode", "window")),
        p_window_swap=p_window_swap,
        p_window_insert=p_window_insert,
        p_window_reorder=p_window_reorder,
        p_window_buffer=p_window_buffer,
        p_window_machine=p_window_machine,
        shift_step=config.get("shift_step"),
        min_block_len=config.get("min_block_len"),
        group_buffer_ratio=config.get("group_buffer_ratio"),
        high_k=config.get("high_k"),
        low_k=config.get("low_k"),
        machine_k=config.get("machine_k"),
        buffer_shift_prob=config.get("scout_buffer_shift_prob", config.get("buffer_shift_prob", 0.25)),
    )


def build_runtime_switches(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "enable_cache": bool(config.get("enable_cache", True)),
        "skip_duplicate_neighbors": bool(config.get("skip_duplicate_neighbors", True)),
        "comfort_scale": float(config.get("comfort_scale", 10.0)),
        "comfort_tau": float(config.get("comfort_tau", 1.0)),
        "init_seed_mode": str(config.get("init_seed_mode", "rng")),
        "init_mode": str(config.get("init_mode", "best")),
        "init_mix_weights": config.get("init_mix_weights"),
        "compare_mode": str(config.get("compare_mode", "lex")),
        "use_pathrisk_tiebreak": bool(config.get("use_pathrisk_tiebreak", True)),
        "two_stage": bool(config.get("two_stage", True)),
        "employ_mode": str(config.get("employ_mode", "risk")),
        "use_risk_elite": bool(config.get("use_risk_elite", True)),
        "use_risk_elite_onl": bool(config.get("use_risk_elite_onl", False)),
        "onlooker_mode": str(config.get("onlooker_mode", "risk")),
        "use_pathrisk_scout": bool(config.get("use_pathrisk_scout", True)),
        "scout_reinit_mode": str(config.get("scout_reinit_mode", "random_topo")),
        "scout_reinit_buffer_mode": str(config.get("scout_reinit_buffer_mode", "random")),
        "two_stage_selector_mode": str(config.get("two_stage_selector_mode", "llm")),
        "buffer_init_mode": str(config.get("buffer_init_mode", "keyness")),
        "llm_local_model_path": str(config.get("llm_local_model_path", "")),
        "llm_two_stage_mode": str(config.get("llm_two_stage_mode", "topk_score")),
        "llm_verify_all_candidates": bool(config.get("llm_verify_all_candidates", False)),
        "llm_verify_pairwise": bool(config.get("llm_verify_pairwise", False)),
        "llm_max_retries": int(config.get("llm_max_retries", 2)),
        "llm_retry_sleep_sec": float(config.get("llm_retry_sleep_sec", 2.0)),
        "llm_fallback_on_error": bool(config.get("llm_fallback_on_error", True)),
    }


def resolve_generated_solutions_limit(config: dict[str, Any], specific_key: str | None = None) -> int | None:
    value = config.get("max_generated_solutions")
    if value is not None:
        return int(value)
    if specific_key is not None:
        value = config.get(specific_key)
        if value is not None:
            return int(value)
    return None


def mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def mean_ignore_none(values: list[float | None]) -> float | None:
    filtered = [value for value in values if value is not None]
    return statistics.mean(filtered) if filtered else None


def mean_positive_only(values: list[float | None]) -> float | None:
    filtered = [value for value in values if value is not None and value > 0]
    return statistics.mean(filtered) if filtered else None


def stdev_or_zero(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _finite_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        f = float(value)
        if math.isfinite(f):
            return f
    except Exception:
        pass
    return None


def parse_seed_list(seed_list_text: str | None) -> list[int]:
    if not seed_list_text:
        return []

    seeds: list[int] = []
    for chunk in seed_list_text.split(","):
        token = chunk.strip()
        if not token:
            continue
        try:
            seeds.append(int(token))
        except Exception as exc:
            raise ValueError(f"invalid seed value in --seed-list: {token!r}") from exc
    return seeds


def compute_keyness_buffer_stats(problem, sol) -> dict[str, float]:
    buf_map = getattr(sol, "buffer", {}) or {}
    keyness_map = {
        uid: safe_float(getattr(op, "keyness_score", 0.0), 0.0)
        for uid, op in problem.ops.items()
    }
    total_alloc = sum(max(0.0, safe_float(v)) for v in buf_map.values())
    if total_alloc <= 1e-12:
        return {
            "KBR": 0.0,
            "high_keyness_avg_buffer": 0.0,
            "low_keyness_avg_buffer": 0.0,
        }

    sorted_uids = sorted(problem.ops.keys(), key=lambda uid: keyness_map.get(uid, 0.0), reverse=True)
    split_idx = max(1, math.ceil(0.3 * len(sorted_uids)))
    high_uids = sorted_uids[:split_idx]
    low_uids = sorted_uids[split_idx:]

    high_total = sum(max(0.0, safe_float(buf_map.get(uid, 0.0))) for uid in high_uids)
    high_avg = high_total / max(1, len(high_uids))
    low_avg = sum(max(0.0, safe_float(buf_map.get(uid, 0.0))) for uid in low_uids) / max(1, len(low_uids))

    return {
        "KBR": high_total / total_alloc,
        "high_keyness_avg_buffer": high_avg,
        "low_keyness_avg_buffer": low_avg,
    }


def compute_pressure_index(problem, sol) -> float:
    buf_map = getattr(sol, "buffer", {}) or {}
    pressure = 0.0
    for uid, op in problem.ops.items():
        ideal_time = max(0.0, safe_float(getattr(op, "ideal_time", 0.0), 0.0))
        buffer = max(0.0, safe_float(buf_map.get(uid, 0.0), 0.0))
        denom = ideal_time + buffer
        if denom <= 1e-12:
            continue
        pressure += safe_float(getattr(op, "keyness_score", 0.0), 0.0) * (ideal_time / denom)
    return pressure


def build_flat_summary(
    *,
    experiment_type: str,
    experiment_name: str,
    run_id: str,
    instance_id: str,
    algorithm: str,
    variant: str,
    seed: int,
    config: dict[str, Any],
    dispatcher: ABCDispatcher,
    best_src,
    problem,
    budgets,
    runtime: float,
) -> dict[str, Any]:
    report = build_run_report(
        best_src=best_src,
        dispatcher=dispatcher,
        problem=problem,
        budgets=budgets,
        config={
            "experiment_type": experiment_type,
            "experiment_name": experiment_name,
            "run_id": run_id,
            "instance_id": instance_id,
            "algorithm": algorithm,
            "variant": variant,
            "seed": seed,
            "llm_strategy": config.get("llm_two_stage_mode", "none"),
            "compare_m": config.get("compare_mode", "lex"),
            "init_method": config.get("init_mode", "best"),
            "scout_strategy": "repair_worth" if config.get("use_pathrisk_scout", True) else "none",
            "buffer_method": str(config.get("buffer_method", "default")),
            "buffer_search_enabled": bool(config.get("buffer_search_enabled", True)),
            "KBR": config.get("KBR", None),
        },
        search_runtime=runtime,
    )

    metrics = getattr(dispatcher, "metrics", {}) or {}
    snapshots = list(metrics.get("iteration_snapshots", []) or [])
    last_snapshot = snapshots[-1] if snapshots else {}
    init_snapshot = snapshots[0] if snapshots else {}

    final_best_z = safe_float(report.get("objective", 0.0), 0.0)
    best_found_iter = None
    best_found_decode_count = None
    for snap in snapshots:
        if abs(safe_float(snap.get("current_best_Z"), 0.0) - final_best_z) <= 1e-9:
            best_found_iter = int(snap.get("iteration", 0) or 0)
            best_found_decode_count = int(snap.get("full_decode_count", 0) or 0)
            break

    keyness_buffer_stats = compute_keyness_buffer_stats(problem, best_src.sol)

    raw_metrics = report.get("raw_metrics", {}) if isinstance(report.get("raw_metrics"), dict) else {}
    total_buffer_alloc = safe_float(report.get("buffer_total_alloc", 0.0), 0.0)
    total_buffer_avail = safe_float(report.get("buffer_total_avail", 0.0), 0.0)
    buffer_utilization = safe_float(report.get("buffer_usage_ratio", 0.0), 0.0)

    dispatcher_max_iters = int(
        getattr(dispatcher, "max_iters", getattr(getattr(dispatcher, "config", None), "max_iters", 0)) or 0
    )

    full_decode_count = int(raw_metrics.get("evals", report.get("evals", 0)) or 0)
    # generated_solution_count now represents the number of generated solution attempts
    # (counts every generation, including duplicates). Prefer `eval_calls` when available.
    generated_unique_count = int(
        metrics.get("eval_calls", raw_metrics.get("evals", 0)) or 0
    )

    summary = {
        "experiment_type": experiment_type,
        "run_id": run_id,
        "instance_id": instance_id,
        "algorithm": algorithm,
        "variant": variant,
        "seed": seed,
        "best_obj": safe_float(report.get("objective", 0.0), 0.0),
        "best_T": safe_float(report.get("total_tardiness", 0.0), 0.0),
        "best_C": safe_float(report.get("comfort", 0.0), 0.0),
        "total_buffer": total_buffer_alloc,
        "buffer_utilization": buffer_utilization,
        "runtime": runtime,
        "full_decode_count": full_decode_count,
        # generated_solution_count: number of unique decoded solutions seen during the run.
        "generated_solution_count": generated_unique_count,
        "best_found_iteration": best_found_iter,
        "best_found_decode_count": best_found_decode_count,
        "final_iteration": int(last_snapshot.get("iteration", dispatcher_max_iters) or dispatcher_max_iters),
        "llm_strategy": config.get("llm_two_stage_mode", "none") if config.get("llm_local_model_path", "") else "none",
        "compare_m": config.get("compare_mode", "lex"),
        "employ_mode": config.get("employ_mode", "risk"),
        "use_risk_elite": config.get("use_risk_elite", True),
        "onlooker_mode": config.get("onlooker_mode", "risk"),
        "candidate_before_screening": int(
            (metrics.get("stage1_unique_emp", 0) or 0)
            + (metrics.get("stage1_unique_onl", 0) or 0)
        ),
        "candidate_after_screening": int(raw_metrics.get("evals", report.get("evals", 0)) or 0),
        "llm_call_count": int(raw_metrics.get("llm_calls", report.get("llm_calls", 0)) or 0),
        "llm_parse_fallbacks": int(raw_metrics.get("llm_parse_fallbacks", report.get("llm_parse_fallbacks", 0)) or 0),
        "llm_choice_a_count": int(raw_metrics.get("llm_choice_a_count", report.get("llm_choice_a_count", 0)) or 0),
        "llm_choice_b_count": int(raw_metrics.get("llm_choice_b_count", report.get("llm_choice_b_count", 0)) or 0),
        "llm_choice_a_rate": safe_float(report.get("llm_choice_a_rate", 0.0), 0.0),
        "llm_choice_b_rate": safe_float(report.get("llm_choice_b_rate", 0.0), 0.0),
        "llm_verify_cases": int(raw_metrics.get("llm_verify_cases", report.get("llm_verify_cases", 0)) or 0),
        "llm_verify_hit_rate": safe_float(report.get("llm_verify_hit_rate", 0.0), 0.0),
        "llm_topk_overlap_ratio": safe_float(report.get("llm_topk_overlap_ratio", 0.0), 0.0),
        "llm_pairwise_hit_rate": safe_float(report.get("llm_pairwise_hit_rate", 0.0), 0.0),
        "init_method": config.get("init_mode", "best"),
        "initial_best_Z": safe_float(init_snapshot.get("current_best_Z", 0.0), 0.0),
        "initial_avg_Z": safe_float(init_snapshot.get("population_avg_Z", 0.0), 0.0),
        "initial_std_Z": safe_float(init_snapshot.get("population_std_Z", 0.0), 0.0),
        "initial_best_TT": safe_float(init_snapshot.get("current_best_T", 0.0), 0.0),
        "initial_best_U": safe_float(init_snapshot.get("current_best_U", 0.0), 0.0),
        "initial_KBR": safe_float(metrics.get("initial_kbr_mean", 0.0), 0.0),
        "scout_strategy": "repair_worth" if config.get("use_pathrisk_scout", True) else "none",
        "scout_activation_count": int(metrics.get("sco_events", 0) or 0),
        "scout_success_iteration_count": int(metrics.get("sco_success_events", 0) or 0),
        "scout_triggered_solution_count": int(metrics.get("sco_triggered_sources", 0) or 0),
        "pathrisk_repair_count": int(metrics.get("pr_calls", 0) or 0),
        "repair_worth_check_count": int(metrics.get("repair_worth_check_count", 0) or 0),
        "repair_worth_accept_count": int(metrics.get("repair_worth_accept_count", 0) or 0),
        "repair_worth_reject_count": int(metrics.get("repair_worth_reject_count", 0) or 0),
        "repair_worth_accept_rate": (
            int(metrics.get("repair_worth_accept_count", 0) or 0)
            / int(metrics.get("repair_worth_check_count", 0) or 0)
            if int(metrics.get("repair_worth_check_count", 0) or 0) > 0
            else None
        ),
        "scout_early_accept_count": int(metrics.get("scout_early_accept_count", 0) or 0),
        "scout_tardy_promote_count": int(metrics.get("sco_tardy_promote_calls", 0) or 0),
        "scout_tardy_promote_success_count": int(metrics.get("sco_tardy_promote_success", 0) or 0),
        "random_restart_count": int(metrics.get("sco_events", 0) or 0),
        "successful_recovery_count": int(metrics.get("sco_success", 0) or 0),
        "recovery_rate": (
            int(metrics.get("sco_success", 0) or 0) / int(metrics.get("pr_calls", 0) or 0)
            if int(metrics.get("pr_calls", 0) or 0) > 0
            else None
        ),
        "scout_improvement_sum": safe_float(metrics.get("sco_improvement_sum", 0.0), 0.0),
        "scout_avg_improvement": (
            safe_float(metrics.get("sco_improvement_sum", 0.0), 0.0)
            / int(metrics.get("sco_success", 0) or 0)
            if int(metrics.get("sco_success", 0) or 0) > 0
            else None
        ),
        "scout_window_detect_count": int(metrics.get("scout_window_detect_count", 0) or 0),
        "scout_window_op_count": int(metrics.get("scout_window_op_count", 0) or 0),
        "scout_window_len_sum": int(metrics.get("scout_window_len_sum", 0) or 0),
        "scout_window_len_max": int(metrics.get("scout_window_len_max", 0) or 0),
        "buffer_method": str(config.get("buffer_method", "default")),
        "KBR": keyness_buffer_stats["KBR"],
        "pressure_index": compute_pressure_index(problem, best_src.sol),
        "high_keyness_avg_buffer": keyness_buffer_stats["high_keyness_avg_buffer"],
        "low_keyness_avg_buffer": keyness_buffer_stats["low_keyness_avg_buffer"],
        "buffer_search_enabled": bool(config.get("buffer_search_enabled", True)),
    }

    return summary


def _build_convergence_rows_from_trace(trace_path: Path, sample_step: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if sample_step <= 0 or not trace_path.exists():
        return rows

    best_so_far = float("inf")
    seen_evals = 0
    next_cutoff = int(sample_step)

    with trace_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            try:
                record = json.loads(line)
            except Exception:
                continue

            objective = _finite_float(record.get("objective"))
            if objective is None:
                continue

            eval_count = record.get("eval_count")
            if isinstance(eval_count, (int, float)) and math.isfinite(float(eval_count)):
                seen_evals = max(seen_evals, int(eval_count))
            else:
                seen_evals += 1

            if objective < best_so_far:
                best_so_far = objective

            while seen_evals >= next_cutoff:
                rows.append(
                    {
                        "full_decode_count": int(next_cutoff),
                        "current_best_Z": round(float(best_so_far), 6),
                    }
                )
                next_cutoff += int(sample_step)

    if not rows and seen_evals > 0 and math.isfinite(best_so_far):
        rows.append(
            {
                "full_decode_count": int(seen_evals),
                "current_best_Z": round(float(best_so_far), 6),
            }
        )

    return rows


def _build_convergence_rows_from_snapshots(snapshots: list[dict[str, Any]], sample_step: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if sample_step <= 0 or not snapshots:
        return rows

    best_so_far = float("inf")
    next_cutoff = int(sample_step)

    ordered = sorted(snapshots, key=lambda snap: int(snap.get("full_decode_count", 0) or 0))
    for snap in ordered:
        full_decode_count = int(snap.get("full_decode_count", 0) or 0)
        current_best = _finite_float(snap.get("current_best_Z"))
        if current_best is not None and current_best < best_so_far:
            best_so_far = current_best

        while full_decode_count >= next_cutoff:
            if math.isfinite(best_so_far):
                rows.append(
                    {
                        "full_decode_count": int(next_cutoff),
                        "current_best_Z": round(float(best_so_far), 6),
                    }
                )
            next_cutoff += int(sample_step)

    if not rows and ordered:
        last_count = int(ordered[-1].get("full_decode_count", 0) or 0)
        last_best = _finite_float(ordered[-1].get("current_best_Z"))
        if last_best is not None:
            rows.append(
                {
                    "full_decode_count": last_count,
                    "current_best_Z": round(float(last_best), 6),
                }
            )

    return rows


def aggregate_convergence_curve(per_run_rows: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    bucketed: dict[int, list[float]] = {}
    for rows in per_run_rows:
        for row in rows:
            count = int(row.get("full_decode_count", 0) or 0)
            best = _finite_float(row.get("current_best_Z"))
            if best is None:
                continue
            bucketed.setdefault(count, []).append(best)

    curve: list[dict[str, Any]] = []
    for count in sorted(bucketed):
        values = bucketed[count]
        curve.append(
            {
                "full_decode_count": count,
                "current_best_Z_mean": mean_or_none(values),
                "current_best_Z_stdev": stdev_or_zero(values),
                "n_runs": len(values),
            }
        )
    return curve


def write_convergence_log(
    *,
    output_path: Path,
    experiment_type: str,
    experiment_name: str,
    run_id: str,
    instance_id: str,
    algorithm: str,
    variant: str,
    seed: int,
    snapshots: list[dict[str, Any]],
    log_step: int,
    decode_sample_step: int | None = None,
    raw_trace_path: Path | None = None,
) -> list[dict[str, Any]]:
    if decode_sample_step is not None:
        if raw_trace_path is not None and raw_trace_path.exists():
            sampled = _build_convergence_rows_from_trace(raw_trace_path, int(decode_sample_step))
        else:
            sampled = _build_convergence_rows_from_snapshots(snapshots, int(decode_sample_step))
    else:
        sampled = [snap for snap in snapshots if snap.get("iteration", 0) == 0 or int(snap.get("iteration", 0) or 0) % log_step == 0]
        if snapshots:
            final_iter = int(snapshots[-1].get("iteration", 0) or 0)
            if sampled and int(sampled[-1].get("iteration", 0) or 0) != final_iter:
                sampled.append(snapshots[-1])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for snap in sampled:
            if "current_best_Z" in snap and "full_decode_count" in snap:
                row = {
                    "full_decode_count": int(snap["full_decode_count"]),
                    "current_best_Z": round(float(snap["current_best_Z"]), 6),
                }
            else:
                row = {
                    "full_decode_count": int(snap.get("full_decode_count", 0) or 0),
                    "current_best_Z": round(float(snap.get("current_best_Z", 0.0)), 6),
                }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    return sampled


def aggregate_batch_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    def values(key: str) -> list[float]:
        out = []
        for record in records:
            value = record.get(key)
            if isinstance(value, (int, float)) and value == value:
                out.append(float(value))
        return out

    def group_stats(key: str) -> dict[str, Any]:
        vals = values(key)
        return {
            "mean": mean_or_none(vals),
            "stdev": stdev_or_zero(vals),
            "min": min(vals) if vals else None,
            "max": max(vals) if vals else None,
        }

    return {
        "n_runs": len(records),
        "best_obj": group_stats("best_obj"),
        "best_T": group_stats("best_T"),
        "best_C": group_stats("best_C"),
        "runtime": group_stats("runtime"),
        "full_decode_count": group_stats("full_decode_count"),
        "generated_solution_count": group_stats("generated_solution_count"),
        "llm_call_count": group_stats("llm_call_count"),
        "llm_choice_a_count": group_stats("llm_choice_a_count"),
        "llm_choice_b_count": group_stats("llm_choice_b_count"),
        "llm_choice_a_rate": group_stats("llm_choice_a_rate"),
        "llm_choice_b_rate": group_stats("llm_choice_b_rate"),
        "scout_activation_count": group_stats("scout_activation_count"),
        "scout_success_iteration_count": group_stats("scout_success_iteration_count"),
        "scout_triggered_solution_count": group_stats("scout_triggered_solution_count"),
        "pathrisk_repair_count": group_stats("pathrisk_repair_count"),
        "repair_worth_check_count": group_stats("repair_worth_check_count"),
        "repair_worth_accept_count": group_stats("repair_worth_accept_count"),
        "repair_worth_reject_count": group_stats("repair_worth_reject_count"),
        "repair_worth_accept_rate": {
            "mean": mean_ignore_none([record.get("repair_worth_accept_rate") for record in records])
        },
        "scout_early_accept_count": group_stats("scout_early_accept_count"),
        "scout_window_detect_count": group_stats("scout_window_detect_count"),
        "scout_window_op_count": group_stats("scout_window_op_count"),
        "scout_window_len_sum": group_stats("scout_window_len_sum"),
        "scout_window_len_max": group_stats("scout_window_len_max"),
        "buffer_utilization": group_stats("buffer_utilization"),
        "successful_recovery_count": group_stats("successful_recovery_count"),
        "recovery_rate": {"mean": mean_positive_only([record.get("recovery_rate") for record in records])},
        "scout_avg_improvement": {"mean": mean_ignore_none([record.get("scout_avg_improvement") for record in records])},
        "initial_best_Z": group_stats("initial_best_Z"),
        "initial_avg_Z": group_stats("initial_avg_Z"),
        "initial_std_Z": group_stats("initial_std_Z"),
        "best_found_iteration": group_stats("best_found_iteration"),
        "initial_KBR": group_stats("initial_KBR"),
        "KBR": group_stats("KBR"),
        "pressure_index": group_stats("pressure_index"),
    }


def load_switch_bundle(path: str | None) -> dict[str, Any]:
    if not path:
        return {"default": {}, "presets": {}}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"switches file not found: {path}")
    data = load_config(path)
    if not isinstance(data, dict):
        raise ValueError("switches file must be a JSON object")
    default = data.get("default", {})
    presets = data.get("presets", {})
    if not isinstance(default, dict) or not isinstance(presets, dict):
        raise ValueError("switches JSON requires object fields: default, presets")
    return {"default": default, "presets": presets}


def resolve_runtime_switches(
    *,
    config: dict[str, Any],
    switch_bundle: dict[str, Any],
    preset_name: str,
) -> dict[str, Any]:
    merged = dict(build_runtime_switches(config))
    merged.update(switch_bundle.get("default", {}))
    preset_overrides = switch_bundle.get("presets", {}).get(preset_name, {})
    if not isinstance(preset_overrides, dict):
        raise ValueError(f"switch preset '{preset_name}' must be an object")
    merged.update(preset_overrides)
    return merged


def history_to_snapshots(
    *,
    history_best_objective: list[float],
    full_decode_count: int,
    generated_solution_count: int,
) -> list[dict[str, Any]]:
    if not history_best_objective:
        return []
    last_idx = len(history_best_objective) - 1
    snaps: list[dict[str, Any]] = []
    for i, z in enumerate(history_best_objective):
        snaps.append(
            {
                "iteration": i,
                "elapsed_time_sec": 0.0,
                "current_best_Z": safe_float(z, 0.0),
                "current_best_T": 0.0,
                "current_best_U": 0.0,
                "population_avg_Z": 0.0,
                "population_std_Z": 0.0,
                "full_decode_count": full_decode_count if i == last_idx else 0,
                "generated_solution_count": generated_solution_count if i == last_idx else 0,
                "llm_call_count": 0,
                "scout_activation_count": 0,
            }
        )
    return snaps


def metrics_to_coarse_snapshots(*, metrics: dict[str, Any], max_iters: int, best_obj: float) -> list[dict[str, Any]]:
    first = safe_float(metrics.get("best_obj_first", best_obj), best_obj)
    mid = safe_float(metrics.get("best_obj_mid", best_obj), best_obj)
    last = safe_float(metrics.get("best_obj_last", best_obj), best_obj)
    rows = [
        {"iteration": 0, "current_best_Z": first, "current_best_T": 0.0, "current_best_U": 0.0},
        {"iteration": max(1, int(max_iters // 2)), "current_best_Z": mid, "current_best_T": 0.0, "current_best_U": 0.0},
        {"iteration": int(max_iters), "current_best_Z": last, "current_best_T": 0.0, "current_best_U": 0.0},
    ]
    return rows


def _proxy_dispatcher_for_baseline(result, *, alpha: float, beta: float, max_iters: int):
    metrics = dict(result.metrics)
    # For baselines, report generated_solution_count as number of generation attempts (`eval_calls`).
    metrics["iteration_snapshots"] = history_to_snapshots(
        history_best_objective=[safe_float(v, 0.0) for v in result.history_best_objective],
        full_decode_count=int(metrics.get("evals", 0) or 0),
        generated_solution_count=int(metrics.get("eval_calls", metrics.get("evals", 0)) or 0),
    )
    return SimpleNamespace(
        metrics=metrics,
        max_iters=max_iters,
        alpha=float(alpha),
        beta=float(beta),
        norm={"tardiness_scale": 1.0, "comfort_scale": 1.0},
        compare_mode="objective",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=None,
        help="algorithm config JSON; defaults to the matching file under data/inputs/configures/repeat/",
    )
    parser.add_argument("--instance", required=True)
    parser.add_argument("--n", type=int, default=20, help="number of random runs")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="base seed for reproducible repeated runs; each run derives a deterministic seed from it",
    )
    parser.add_argument(
        "--seed-list",
        default=None,
        help="comma-separated explicit run seeds; when provided, overrides --seed-derived scheduling",
    )
    parser.add_argument("--log-step", type=int, default=5, help="save convergence logs every N iterations")
    parser.add_argument("--out-dir", default="data/outputs/reports/repeat_runs")
    parser.add_argument(
        "--algorithm",
        default="abc",
        choices=["abc", "abc_baseline", "pso_baseline", "asa_baseline", "sa_baseline", "asa_paper_baseline", "pbiga_adapt", "tama", "hqpso_vns"],
        help="which algorithm family to run repeatedly",
    )
    parser.add_argument(
        "--switches-file",
        default="/data/Wanglele/ABC_schedule/data/inputs/configures/abc_switch_presets.json",
        help="JSON file containing default and preset switch configs for ABC",
    )
    parser.add_argument("--switch-preset", default="A", help="switch preset name from switches file (ABC only)")
    parser.add_argument(
        "--two-stage-selector-mode",
        default=None,
        choices=["llm", "incremental", "full_decode", "random_topk"],
        help="override the two-stage selector mode for ABC",
    )
    parser.add_argument(
        "--buffer-init-mode",
        default=None,
        choices=["keyness", "random", "uniform"],
        help="override the initial buffer allocation mode for ABC",
    )
    parser.add_argument(
        "--decode-sample-step",
        type=int,
        default=None,
        help="sample convergence every N full decodes and write decode-count-based points",
    )
    parser.add_argument("--experiment-type", default="repeat_random")
    parser.add_argument("--experiment-name", default="ABC_repeat")
    parser.add_argument("--variant", default="full")
    parser.add_argument("--save-raw-trace", action="store_true", help="also save raw encoded-solution trace")
    parser.add_argument(
        "--no-save-gantt",
        action="store_true",
        help="do not save the best-solution Gantt chart for each run",
    )
    parser.add_argument(
        "--gantt-no-labels",
        action="store_true",
        help="hide operation labels in saved Gantt charts",
    )
    args = parser.parse_args()

    algo_name = str(args.algorithm).strip().lower()
    if algo_name == "asa_baseline":
        algo_name = "sa_baseline"
    config_path = resolve_config_path(algo_name, args.config)
    config = load_config(str(config_path))
    switch_bundle = load_switch_bundle(args.switches_file)
    switches = resolve_runtime_switches(config=config, switch_bundle=switch_bundle, preset_name=args.switch_preset)
    selector_mode = str(args.two_stage_selector_mode or switches["two_stage_selector_mode"]).strip().lower()
    buffer_init_mode = str(args.buffer_init_mode or switches["buffer_init_mode"]).strip().lower()
    runtime_config = {**config, **switches}
    emp_cfg = make_emp_cfg(runtime_config)
    onl_cfg = make_onl_cfg(runtime_config)
    sco_cfg = make_sco_cfg(config)

    config_tag = make_config_tag(str(config_path))
    instance_number = extract_suffix_number(args.instance)
    instance_id = f"I{instance_number}" if instance_number is not None else Path(args.instance).stem

    problem = SchedulingProblem()
    DataLoader.load_from_json(args.instance, problem)

    # Match the single-run demo pipeline so keyness-dependent logic has
    # the same input state in repeated experiments.
    problem.calculate_time_windows()
    problem.build_conflict_graph()
    problem.calculate_keyness(
        w1=float(config.get("w1", 0.6)),
        w2=float(config.get("w2", 0.4)),
    )

    asap_res = asap_schedule(problem)
    budgets = compute_order_budgets(
        problem,
        asap_res,
        alpha_B=config.get("alpha_B", 1.0),
        reserve_ratio=config.get("reserve_ratio", 0.0),
        bmax_cap_ratio=config.get("bmax_cap_ratio", 0.0),
        buffer_budget_mode=switches.get("buffer_budget_mode", config.get("buffer_budget_mode", "job")),
        global_buffer_ratio=switches.get("global_buffer_ratio", config.get("global_buffer_ratio", 1.0)),
        job_cap_xi=switches.get("job_cap_xi", config.get("job_cap_xi", 2.0)),
        job_cap_gamma=switches.get("job_cap_gamma", config.get("job_cap_gamma", 0.4)),
        job_cap_omega_n=switches.get("job_cap_omega_n", config.get("job_cap_omega_n", 0.2)),
        job_cap_omega_w=switches.get("job_cap_omega_w", config.get("job_cap_omega_w", 0.8)),
    )

    batch_stamp = datetime.now().strftime("%m%d_%H%M%S")
    batch_dir = Path(args.out_dir) / f"{batch_stamp}_{config_tag}_{instance_id}_{args.experiment_name}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    per_run_records: list[dict[str, Any]] = []
    per_run_convergence_rows: list[list[dict[str, Any]]] = []

    explicit_seed_list = parse_seed_list(args.seed_list)
    if explicit_seed_list and len(explicit_seed_list) != args.n:
        raise ValueError(
            f"--seed-list length ({len(explicit_seed_list)}) must match --n ({args.n}) when provided"
        )

    if explicit_seed_list:
        seed_schedule = explicit_seed_list
        seed_source = "explicit_list"
    elif args.seed is not None:
        seed_rng = random.Random(args.seed)
        seed_schedule = [seed_rng.randint(0, 2**31 - 1) for _ in range(args.n)]
        seed_source = "base_seed"
    else:
        seed_rng = random.SystemRandom()
        seed_schedule = [seed_rng.randint(0, 2**31 - 1) for _ in range(args.n)]
        seed_source = "system_random"

    for run_idx in range(1, args.n + 1):
        seed_i = seed_schedule[run_idx - 1]
        run_id = f"run_{run_idx:02d}"
        run_dir = batch_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        raw_trace_path = run_dir / f"{run_id}_trace.jsonl" if args.save_raw_trace else None
        llm_trace_path = run_dir / f"{run_id}_trace.llm.jsonl" if args.save_raw_trace and algo_name == "abc" else None
        if llm_trace_path is not None:
            llm_trace_path.parent.mkdir(parents=True, exist_ok=True)
            llm_trace_path.touch(exist_ok=True)
        train_trace_path = run_dir / f"{run_id}_train.jsonl" if args.save_raw_trace and algo_name == "abc" else None
        summary_obj: dict[str, Any]
        snapshots: list[dict[str, Any]]
        t0 = time.perf_counter()

        if algo_name == "abc":
            abc = ABCDispatcher(
                problem=problem,
                budgets=budgets,
                alpha=config.get("alpha", 1.0),
                beta=config.get("beta", 1.0),
                verbose=False,
                enable_cache=switches["enable_cache"],
                skip_duplicate_neighbors=switches["skip_duplicate_neighbors"],
                compare_mode=switches["compare_mode"],
                use_pathrisk_tiebreak=switches["use_pathrisk_tiebreak"],
                colony_size=config.get("colony_size", 20),
                max_iters=config.get("max_iters", 100),
                limit=config.get("limit", 100),
                seed=seed_i,
                init_mode=switches["init_mode"],
                init_seed_mode=switches["init_seed_mode"],
                rcl_size=config.get("rcl_size", 1),
                rcl_tau=config.get("rcl_tau", 0.0),
                max_candidates=config.get("max_candidates", 20),
                allow_fallback_topo=True,
                max_passes=config.get("max_passes", 3),
                top_risk_due_weight=config.get("top_risk_due_weight", 1.0),
                integer_buffer=config.get("integer_buffer", False),
                buffer_unit=config.get("buffer_unit", 0.5),
                buffer_init_mode=buffer_init_mode,
                emp_cfg=emp_cfg,
                onl_cfg=onl_cfg,
                two_stage=switches["two_stage"],
                use_incremental_rerank=bool(config.get("use_incremental_rerank", False)),
                inc_factor=config.get("inc_factor", 3),
                inc_window=config.get("inc_window", 12),
                inc_succ_hops=config.get("inc_succ_hops", 2),
                inc_max_cover=config.get("inc_max_cover", 0.65),
                inc_diag_enabled=config.get("inc_diag_enabled", False),
                inc_diag_sample_prob=config.get("inc_diag_sample_prob", 0.2),
                inc_diag_max_pool=config.get("inc_diag_max_pool", 8),
                decode_prune_enabled=config.get("decode_prune_enabled", True),
                decode_prune_eps=config.get("decode_prune_eps", 1e-9),
                decode_prune_cert_cache_enabled=config.get("decode_prune_cert_cache_enabled", True),
                M_emp=config.get("M_emp", 10),
                K_emp=config.get("K_emp", 3),
                M_onl=config.get("M_onl", 10),
                K_onl=config.get("K_onl", 3),
                llm_two_stage_mode="off" if selector_mode in ("incremental", "full_decode", "random_topk") else switches["llm_two_stage_mode"],
                two_stage_selector_mode=selector_mode,
                llm_local_model_path=switches["llm_local_model_path"],
                llm_instance_id=instance_id,
                llm_device_map=config.get("llm_device_map", "auto"),
                llm_torch_dtype=config.get("llm_torch_dtype", "auto"),
                llm_max_new_tokens=config.get("llm_max_new_tokens", 4),
                llm_temperature=config.get("llm_temperature", 0.0),
                llm_disable_trust_remote_code=config.get("llm_disable_trust_remote_code", False),
                llm_disable_fast_tokenizer=config.get("llm_disable_fast_tokenizer", False),
                llm_max_retries=switches["llm_max_retries"],
                llm_retry_sleep_sec=switches["llm_retry_sleep_sec"],
                llm_fallback_on_error=switches["llm_fallback_on_error"],
                llm_boundary_width=config.get("llm_boundary_width", 2),
                llm_boundary_rounds=config.get("llm_boundary_rounds", 2),
                llm_score_m=config.get("llm_score_m", 4),
                llm_trace_file_path=str(llm_trace_path) if llm_trace_path else None,
                llm_verify_all_candidates=switches["llm_verify_all_candidates"],
                llm_verify_pairwise=switches["llm_verify_pairwise"],
                use_pathrisk_scout=switches["use_pathrisk_scout"],
                scout_top_frac=config.get("scout_top_frac", 0.0),
                sco_cfg=sco_cfg,
                trace_file_path=str(raw_trace_path) if raw_trace_path else None,
                init_mix_weights=switches.get("init_mix_weights"),
                scout_reinit_mode=str(switches.get("scout_reinit_mode", "random_topo")),
                scout_reinit_buffer_mode=str(switches.get("scout_reinit_buffer_mode", "random")),
                max_generated_solutions=resolve_generated_solutions_limit(config, "abc_max_generated_solutions"),
                onl_potential_weight=float(runtime_config.get("onl_potential_weight", 0.5)),
                employ_mode=str(runtime_config.get("employ_mode", "risk")),
                onlooker_mode=str(runtime_config.get("onlooker_mode", "risk")),
                sa_acceptance_enabled=bool(runtime_config.get("sa_acceptance_enabled", False)),
                sa_init_temp_factor=float(config.get("sa_init_temp_factor", 0.25)),
                sa_final_temp_ratio=float(config.get("sa_final_temp_ratio", 0.02)),
                sa_tardy_worsen_ratio=float(config.get("sa_tardy_worsen_ratio", 0.0)),
                buffer_pso_enabled=bool(config.get("buffer_pso_enabled", False)),
                buffer_pso_mode=str(config.get("buffer_pso_mode", "hybrid")),
                buffer_pso_prob=float(config.get("buffer_pso_prob", 1.0)),
                buffer_pso_w_start=float(config.get("buffer_pso_w_start", 0.8)),
                buffer_pso_w_end=float(config.get("buffer_pso_w_end", 0.3)),
                buffer_pso_c1=float(config.get("buffer_pso_c1", 1.4)),
                buffer_pso_c2=float(config.get("buffer_pso_c2", 1.4)),
                buffer_pso_keyness_weight=float(config.get("buffer_pso_keyness_weight", 0.0)),
                buffer_pso_vmax=float(config.get("buffer_pso_vmax", 0.2)),
                comfort_scale=float(runtime_config.get("comfort_scale", 10.0)),
                comfort_tau=float(runtime_config.get("comfort_tau", 1.0)),
            )
            best_src = abc.run()
            runtime = time.perf_counter() - t0
            summary_obj = build_flat_summary(
                experiment_type=args.experiment_type,
                experiment_name=args.experiment_name,
                run_id=run_id,
                instance_id=instance_id,
                algorithm=algo_name,
                variant=args.variant,
                seed=seed_i,
                config={**config, **switches, "buffer_init_mode": buffer_init_mode, "two_stage_selector_mode": selector_mode},
                dispatcher=abc,
                best_src=best_src,
                problem=problem,
                budgets=budgets,
                runtime=runtime,
            )
            snapshots = list(getattr(abc, "metrics", {}).get("iteration_snapshots", []) or [])

            if train_trace_path is not None and raw_trace_path is not None:
                n_train_rows = export_training_records_from_trace(
                    trace_file_path=str(raw_trace_path),
                    output_file_path=str(train_trace_path),
                    problem=problem,
                    instans=instance_id,
                )
            else:
                n_train_rows = None

        elif algo_name == "abc_baseline":
            baseline = SingleObjectiveABCBaseline(
                problem=problem,
                budgets=budgets,
                alpha=float(config.get("alpha", 1.0)),
                beta=float(config.get("beta", 1.0)),
                colony_size=int(config.get("colony_size", 20)),
                max_iters=int(config.get("max_iters", 50)),
                limit=int(config.get("limit", 20)),
                seed=seed_i,
                integer_buffer=bool(config.get("integer_buffer", False)),
                buffer_unit=float(config.get("buffer_unit", 0.5)),
                trace_file_path=str(raw_trace_path) if raw_trace_path else None,
                enable_cache=bool(config.get("enable_cache", False)),
                max_generated_solutions=resolve_generated_solutions_limit(config, "abc_max_generated_solutions"),
                comfort_scale=float(config.get("comfort_scale", 10.0)),
                comfort_tau=float(config.get("comfort_tau", 1.0)),
            )
            result = baseline.run()
            runtime = float(result.metrics.get("search_runtime", time.perf_counter() - t0))
            best_src = SimpleNamespace(sol=result.best_solution, res=result.best_result)
            proxy = _proxy_dispatcher_for_baseline(
                result,
                alpha=float(config.get("alpha", 1.0)),
                beta=float(config.get("beta", 1.0)),
                max_iters=int(config.get("max_iters", 50)),
            )
            summary_obj = build_flat_summary(
                experiment_type=args.experiment_type,
                experiment_name=args.experiment_name,
                run_id=run_id,
                instance_id=instance_id,
                algorithm=algo_name,
                variant=args.variant,
                seed=seed_i,
                config=config,
                dispatcher=proxy,
                best_src=best_src,
                problem=problem,
                budgets=budgets,
                runtime=runtime,
            )
            snapshots = list(proxy.metrics.get("iteration_snapshots", []) or [])

        elif algo_name == "pso_baseline":
            pso = SingleObjectivePSOBaseline(
                problem=problem,
                budgets=budgets,
                alpha=float(config.get("alpha", 1.0)),
                beta=float(config.get("beta", 1.0)),
                swarm_size=int(config.get("pso_swarm_size", config.get("colony_size", 30))),
                max_iters=int(config.get("pso_max_iters", config.get("max_iters", 80))),
                inertia_start=float(config.get("pso_inertia_start", 0.9)),
                inertia_end=float(config.get("pso_inertia_end", 0.4)),
                c1=float(config.get("pso_c1", 1.8)),
                c2=float(config.get("pso_c2", 1.8)),
                vmax=float(config.get("pso_vmax", 0.2)),
                seed=seed_i,
                integer_buffer=bool(config.get("integer_buffer", False)),
                buffer_unit=float(config.get("buffer_unit", 0.5)),
                trace_file_path=str(raw_trace_path) if raw_trace_path else None,
                enable_cache=bool(config.get("enable_cache", False)),
                max_generated_solutions=resolve_generated_solutions_limit(config),
                comfort_scale=float(config.get("comfort_scale", 10.0)),
                comfort_tau=float(config.get("comfort_tau", 1.0)),
            )
            result = pso.run()
            runtime = float(result.metrics.get("search_runtime", time.perf_counter() - t0))
            best_src = SimpleNamespace(sol=result.best_solution, res=result.best_result)
            proxy = _proxy_dispatcher_for_baseline(
                result,
                alpha=float(config.get("alpha", 1.0)),
                beta=float(config.get("beta", 1.0)),
                max_iters=int(config.get("pso_max_iters", config.get("max_iters", 80))),
            )
            summary_obj = build_flat_summary(
                experiment_type=args.experiment_type,
                experiment_name=args.experiment_name,
                run_id=run_id,
                instance_id=instance_id,
                algorithm=algo_name,
                variant=args.variant,
                seed=seed_i,
                config=config,
                dispatcher=proxy,
                best_src=best_src,
                problem=problem,
                budgets=budgets,
                runtime=runtime,
            )
            snapshots = list(proxy.metrics.get("iteration_snapshots", []) or [])

        elif algo_name == "sa_baseline":
            _raw_max_iters = config.get("asa_max_iters")
            if _raw_max_iters is None:
                _max_iters = int(config.get("max_iters", 80))
            else:
                _max_iters = int(_raw_max_iters)

            asa = SingleObjectiveSABaseline(
                problem=problem,
                budgets=budgets,
                alpha=float(config.get("alpha", 1.0)),
                beta=float(config.get("beta", 1.0)),
                max_iters=_max_iters,
                chain_length=int(config.get("asa_chain_length", 40)),
                init_temp=float(config.get("asa_init_temp", 1.0)),
                final_temp=float(config.get("asa_final_temp", 1e-3)),
                cooling_rate=float(config.get("asa_cooling_rate", 0.95)),
                target_acceptance=float(config.get("asa_target_acceptance", 0.25)),
                adapt_gain=float(config.get("asa_adapt_gain", 0.5)),
                adapt_interval=int(config.get("asa_adapt_interval", 5)),
                reheat_factor=float(config.get("asa_reheat_factor", 1.5)),
                stagnation_patience=int(config.get("asa_stagnation_patience", 12)),
                bns=int(config.get("asa_bns", 4)),
                q_alpha=float(config.get("asa_q_alpha", 0.2)),
                q_gamma=float(config.get("asa_q_gamma", 0.9)),
                q_epsilon=float(config.get("asa_q_epsilon", 0.25)),
                q_epsilon_min=float(config.get("asa_q_epsilon_min", 0.05)),
                q_decay=float(config.get("asa_q_decay", 0.995)),
                init_pool_size=int(config.get("asa_init_pool_size", 6)),
                operator_bonus=float(config.get("asa_operator_bonus", 0.2)),
                seed=seed_i,
                integer_buffer=bool(config.get("integer_buffer", False)),
                buffer_unit=float(config.get("buffer_unit", 0.5)),
                trace_file_path=str(raw_trace_path) if raw_trace_path else None,
                enable_cache=bool(config.get("enable_cache", False)),
                max_generated_solutions=resolve_generated_solutions_limit(config, "asa_max_generated_solutions"),
                comfort_scale=float(config.get("comfort_scale", 10.0)),
                comfort_tau=float(config.get("comfort_tau", 1.0)),
            )
            result = asa.run()
            runtime = float(result.metrics.get("search_runtime", time.perf_counter() - t0))
            best_src = SimpleNamespace(sol=result.best_solution, res=result.best_result)
            proxy = _proxy_dispatcher_for_baseline(
                result,
                alpha=float(config.get("alpha", 1.0)),
                beta=float(config.get("beta", 1.0)),
                max_iters=_max_iters,
            )
            summary_obj = build_flat_summary(
                experiment_type=args.experiment_type,
                experiment_name=args.experiment_name,
                run_id=run_id,
                instance_id=instance_id,
                algorithm=algo_name,
                variant=args.variant,
                seed=seed_i,
                config=config,
                dispatcher=proxy,
                best_src=best_src,
                problem=problem,
                budgets=budgets,
                runtime=runtime,
            )
            snapshots = list(proxy.metrics.get("iteration_snapshots", []) or [])

        elif algo_name == "asa_paper_baseline":
            _raw_max_iters = config.get("asa_max_iters")
            if _raw_max_iters is None:
                _max_iters = int(config.get("max_iters", 80))
            else:
                _max_iters = int(_raw_max_iters)

            asa = SingleObjectiveASAPaperBaseline(
                problem=problem,
                budgets=budgets,
                alpha=float(config.get("alpha", 1.0)),
                beta=float(config.get("beta", 1.0)),
                max_iters=_max_iters,
                chain_length=int(config.get("asa_chain_length", 40)),
                init_temp=float(config.get("asa_init_temp", 1.0)),
                final_temp=float(config.get("asa_final_temp", 1e-3)),
                cooling_rate=float(config.get("asa_cooling_rate", 0.95)),
                target_acceptance=float(config.get("asa_target_acceptance", 0.25)),
                adapt_gain=float(config.get("asa_adapt_gain", 0.5)),
                adapt_interval=int(config.get("asa_adapt_interval", 5)),
                reheat_factor=float(config.get("asa_reheat_factor", 1.5)),
                stagnation_patience=int(config.get("asa_stagnation_patience", 12)),
                bns=int(config.get("asa_bns", 4)),
                q_alpha=float(config.get("asa_q_alpha", 0.2)),
                q_gamma=float(config.get("asa_q_gamma", 0.9)),
                q_epsilon=float(config.get("asa_q_epsilon", 0.25)),
                q_epsilon_min=float(config.get("asa_q_epsilon_min", 0.05)),
                q_decay=float(config.get("asa_q_decay", 0.995)),
                init_pool_size=int(config.get("asa_init_pool_size", 6)),
                operator_bonus=float(config.get("asa_operator_bonus", 0.2)),
                seed=seed_i,
                integer_buffer=bool(config.get("integer_buffer", False)),
                buffer_unit=float(config.get("buffer_unit", 0.5)),
                trace_file_path=str(raw_trace_path) if raw_trace_path else None,
                enable_cache=bool(config.get("enable_cache", False)),
                max_generated_solutions=resolve_generated_solutions_limit(config, "asa_max_generated_solutions"),
                paper_strict=bool(config.get("asa_paper_strict", False)),
                comfort_scale=float(config.get("comfort_scale", 10.0)),
                comfort_tau=float(config.get("comfort_tau", 1.0)),
            )
            result = asa.run()
            runtime = float(result.metrics.get("search_runtime", time.perf_counter() - t0))
            best_src = SimpleNamespace(sol=result.best_solution, res=result.best_result)
            proxy = _proxy_dispatcher_for_baseline(
                result,
                alpha=float(config.get("alpha", 1.0)),
                beta=float(config.get("beta", 1.0)),
                max_iters=_max_iters,
            )
            summary_obj = build_flat_summary(
                experiment_type=args.experiment_type,
                experiment_name=args.experiment_name,
                run_id=run_id,
                instance_id=instance_id,
                algorithm=algo_name,
                variant=args.variant,
                seed=seed_i,
                config=config,
                dispatcher=proxy,
                best_src=best_src,
                problem=problem,
                budgets=budgets,
                runtime=runtime,
            )
            snapshots = list(proxy.metrics.get("iteration_snapshots", []) or [])

        elif algo_name == "pbiga_adapt":
            pbiga_cfg = PBIGAAdaptConfig(
                population_size=int(config.get("pbiga_population_size", 60)),
                max_iters=int(config.get("pbiga_max_iters", 50)),
                alpha=float(config.get("pbiga_alpha", 0.7)),
                beta=float(config.get("pbiga_beta", 0.7)),
                comfort_scale=float(config.get("comfort_scale", 10.0)),
                comfort_tau=float(config.get("comfort_tau", 1.0)),
                seed=seed_i,
                epsilon=float(config.get("pbiga_epsilon", 0.25)),
                epsilon_decay=float(config.get("pbiga_epsilon_decay", 0.995)),
                epsilon_min=float(config.get("pbiga_epsilon_min", 0.05)),
                q_alpha=float(config.get("pbiga_q_alpha", 0.2)),
                q_gamma=float(config.get("pbiga_q_gamma", 0.9)),
                archive_limit=int(config.get("pbiga_archive_limit", 24)),
                restart_patience=int(config.get("pbiga_restart_patience", 3)),
                destroy_ratio=float(config.get("pbiga_destroy_ratio", 0.18)),
                integer_buffer=bool(config.get("integer_buffer", False)),
                buffer_unit=float(config.get("buffer_unit", 0.5)),
                trace_file_path=str(raw_trace_path) if raw_trace_path else None,
                init_modes=tuple(config.get("pbiga_init_modes", ("best", "random", "topo"))),
                rcl_size=int(config.get("pbiga_rcl_size", 3)),
                rcl_tau=float(config.get("pbiga_rcl_tau", 0.0)),
                max_candidates=int(config.get("pbiga_max_candidates", 30)),
                max_passes=int(config.get("pbiga_max_passes", 3)),
                allow_fallback_topo=bool(config.get("pbiga_allow_fallback_topo", True)),
                top_risk_due_weight=float(config.get("pbiga_top_risk_due_weight", 1.0)),
                compare_mode=str(config.get("pbiga_compare_mode", "pareto")),
                use_pathrisk_tiebreak=bool(config.get("pbiga_use_pathrisk_tiebreak", False)),
                use_pathrisk_scout=bool(config.get("pbiga_use_pathrisk_scout", True)),
            )
            pbiga = PBIGAAdaptDispatcher(
                problem=problem,
                budgets=budgets,
                alpha=float(config.get("alpha", 1.0)),
                beta=float(config.get("beta", 1.0)),
                config=pbiga_cfg,
            )
            result = pbiga.run()
            runtime = time.perf_counter() - t0
            if result.get("best_solution") is None or result.get("best_result") is None:
                raise RuntimeError("PBIGA produced no best solution")
            best_src = SimpleNamespace(sol=result["best_solution"], res=result["best_result"])
            pbiga.metrics["iteration_snapshots"] = metrics_to_coarse_snapshots(
                metrics=pbiga.metrics,
                max_iters=pbiga_cfg.max_iters,
                best_obj=safe_float(result["best_result"].objective, 0.0),
            )
            summary_obj = build_flat_summary(
                experiment_type=args.experiment_type,
                experiment_name=args.experiment_name,
                run_id=run_id,
                instance_id=instance_id,
                algorithm=algo_name,
                variant=args.variant,
                seed=seed_i,
                config=config,
                dispatcher=pbiga,
                best_src=best_src,
                problem=problem,
                budgets=budgets,
                runtime=runtime,
            )
            snapshots = list(pbiga.metrics.get("iteration_snapshots", []) or [])

        elif algo_name == "hqpso_vns":
            # HQPSO-VNS single-run wrapper (the run_hqpso_vns function handles reporting)
            # ensure seed passed in config
            config["seed"] = seed_i
            config["trace_file_path"] = str(raw_trace_path) if raw_trace_path else None
            t0_algo = time.perf_counter()
            result = run_hqpso_vns(
                config=config,
                problem=problem,
                budgets=budgets,
                config_number=config_tag,
                instance_number=instance_number,
            )
            runtime = time.perf_counter() - t0_algo
            if result.get("best_solution") is None or result.get("best_result") is None:
                raise RuntimeError("HQPSO-VNS produced no best solution")
            best_src = SimpleNamespace(sol=result["best_solution"], res=result["best_result"])
            # build a lightweight dispatcher proxy for summary generation
            proxy = SimpleNamespace()
            proxy.metrics = result.get("metrics", {}) or {}
            proxy.metrics["iteration_snapshots"] = metrics_to_coarse_snapshots(
                metrics=proxy.metrics,
                max_iters=int(config.get("hqpso_max_iters", config.get("max_iters", 500))),
                best_obj=safe_float(best_src.res.objective, 0.0),
            )
            proxy.max_iters = int(config.get("hqpso_max_iters", config.get("max_iters", 500)))
            proxy.norm = {"tardiness_scale": 1.0, "comfort_scale": 1.0}
            summary_obj = build_flat_summary(
                experiment_type=args.experiment_type,
                experiment_name=args.experiment_name,
                run_id=run_id,
                instance_id=instance_id,
                algorithm=algo_name,
                variant=args.variant,
                seed=seed_i,
                config=config,
                dispatcher=proxy,
                best_src=best_src,
                problem=problem,
                budgets=budgets,
                runtime=runtime,
            )
            snapshots = list(proxy.metrics.get("iteration_snapshots", []) or [])

        elif algo_name == "tama":
            tama_cfg = TAMAConfig(
                population_size=int(config.get("tama_population_size", 300)),
                max_iters=int(config.get("tama_max_iters", 100)),
                alpha=float(config.get("tama_alpha", 1.0)),
                beta=float(config.get("tama_beta", 1.0)),
                comfort_scale=float(config.get("comfort_scale", 10.0)),
                comfort_tau=float(config.get("comfort_tau", 1.0)),
                seed=seed_i,
                epsilon=float(config.get("tama_epsilon", 0.85)),
                epsilon_decay=float(config.get("tama_epsilon_decay", 0.995)),
                epsilon_min=float(config.get("tama_epsilon_min", 0.05)),
                gamma=float(config.get("tama_gamma", 0.9)),
                sarsa_alpha=float(config.get("tama_sarsa_alpha", 0.045)),
                archive_limit=int(config.get("tama_archive_limit", 24)),
                verbose=False,
                integer_buffer=bool(config.get("integer_buffer", False)),
                buffer_unit=float(config.get("buffer_unit", 0.5)),
                trace_file_path=str(raw_trace_path) if raw_trace_path else None,
                init_modes=tuple(config.get("tama_init_modes", ("best", "random", "topo"))),
                top_risk_due_weight=float(config.get("tama_top_risk_due_weight", 1.0)),
                rcl_size=int(config.get("tama_rcl_size", 3)),
                rcl_tau=float(config.get("tama_rcl_tau", 0.0)),
                max_candidates=int(config.get("tama_max_candidates", 30)),
                max_passes=int(config.get("tama_max_passes", 3)),
                allow_fallback_topo=bool(config.get("tama_allow_fallback_topo", True)),
                compare_mode=str(config.get("tama_compare_mode", "objective")),
            )
            tama = TAMADispatcher(
                problem=problem,
                budgets=budgets,
                alpha=float(config.get("alpha", 1.0)),
                beta=float(config.get("beta", 1.0)),
                config=tama_cfg,
            )
            result = tama.run()
            runtime = time.perf_counter() - t0
            if result.get("best_solution") is None or result.get("best_result") is None:
                raise RuntimeError("TAMA produced no best solution")
            best_src = SimpleNamespace(sol=result["best_solution"], res=result["best_result"])
            tama.metrics["iteration_snapshots"] = metrics_to_coarse_snapshots(
                metrics=tama.metrics,
                max_iters=tama_cfg.max_iters,
                best_obj=safe_float(result["best_result"].objective, 0.0),
            )
            summary_obj = build_flat_summary(
                experiment_type=args.experiment_type,
                experiment_name=args.experiment_name,
                run_id=run_id,
                instance_id=instance_id,
                algorithm=algo_name,
                variant=args.variant,
                seed=seed_i,
                config=config,
                dispatcher=tama,
                best_src=best_src,
                problem=problem,
                budgets=budgets,
                runtime=runtime,
            )
            snapshots = list(tama.metrics.get("iteration_snapshots", []) or [])

        else:
            raise ValueError(f"Unsupported algorithm: {algo_name}")

        gantt_path = None
        if not args.no_save_gantt:
            gantt_path = run_dir / f"{run_id}_gantt.png"
            try:
                job_urgency = _compute_job_urgency(problem, best_src.sol.buffer)
                high_risk_path = find_top_risk_path(
                    problem,
                    job_urgency,
                    due_weight=float(config.get("top_risk_due_weight", 1.0)),
                )
            except Exception:
                high_risk_path = None
            plot_gantt(
                problem,
                best_src.res,
                title=f"{algo_name} {instance_id} {run_id} Best Schedule",
                top_risk_path=high_risk_path,
                buffer_map=best_src.sol.buffer,
                show_buffer_split=True,
                show_op_label=not args.gantt_no_labels,
                save_path=str(gantt_path),
            )

        summary_path = run_dir / f"{run_id}_summary.json"
        if gantt_path is not None:
            summary_obj["gantt_file"] = str(gantt_path)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary_obj, f, ensure_ascii=False, indent=2)

        convergence_path = run_dir / f"{run_id}_convergence.jsonl"
        convergence_rows = write_convergence_log(
            output_path=convergence_path,
            experiment_type=args.experiment_type,
            experiment_name=args.experiment_name,
            run_id=run_id,
            instance_id=instance_id,
            algorithm=algo_name,
            variant=args.variant,
            seed=seed_i,
            snapshots=snapshots,
            log_step=max(1, int(args.log_step)),
            decode_sample_step=args.decode_sample_step,
            raw_trace_path=raw_trace_path,
        )
        per_run_convergence_rows.append(convergence_rows)

        # ensure n_train_rows is defined in all branches
        if "n_train_rows" not in locals():
            n_train_rows = None

        manifest_record = {
            "run_id": run_id,
            "seed": seed_i,
            "summary_file": str(summary_path),
            "convergence_file": str(convergence_path),
            "raw_trace_file": str(raw_trace_path) if raw_trace_path else None,
            "llm_trace_file": str(llm_trace_path) if llm_trace_path else None,
            "train_trace_file": str(train_trace_path) if train_trace_path else None,
            "gantt_file": str(gantt_path) if gantt_path is not None else None,
            "train_rows": n_train_rows,
            "convergence_rows": len(convergence_rows),
            "buffer_init_mode": buffer_init_mode,
            "two_stage_selector_mode": selector_mode,
            "employ_mode": switches.get("employ_mode"),
            "onlooker_mode": switches.get("onlooker_mode"),
            "best_obj": summary_obj.get("best_obj", None),
            "best_T": summary_obj.get("best_T", None),
            "best_C": summary_obj.get("best_C", None),
            "runtime": summary_obj.get("runtime", None),
            "full_decode_count": summary_obj.get("full_decode_count", None),
            "generated_solution_count": summary_obj.get("generated_solution_count", None),
            "llm_call_count": summary_obj.get("llm_call_count", None),
            "llm_verify_hit_rate": summary_obj.get("llm_verify_hit_rate", None),
            "llm_topk_overlap_ratio": summary_obj.get("llm_topk_overlap_ratio", None),
            "llm_pairwise_hit_rate": summary_obj.get("llm_pairwise_hit_rate", None),
            "scout_activation_count": summary_obj.get("scout_activation_count", None),
        }
        manifest_rows.append(manifest_record)
        per_run_records.append(summary_obj)

        print(
            f"[{run_id}] seed={seed_i} runtime={runtime:.3f}s "
            f"obj={(summary_obj or {}).get('best_obj', 0.0):.3f} "
            f"summary={summary_path.name} convergence={convergence_path.name}"
        )

    manifest_path = batch_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as fout:
        for row in manifest_rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    batch_summary = {
        "experiment_type": args.experiment_type,
        "experiment_name": args.experiment_name,
        "algorithm": algo_name,
        "variant": args.variant,
        "config_file": str(config_path),
        "instance_file": args.instance,
        "batch_dir": str(batch_dir),
        "n_runs": len(per_run_records),
        "seed_source": seed_source,
        "seed_schedule": seed_schedule,
        "buffer_init_mode": buffer_init_mode,
        "two_stage_selector_mode": selector_mode,
        "employ_mode": switches.get("employ_mode"),
        "onlooker_mode": switches.get("onlooker_mode"),
        "log_step": max(1, int(args.log_step)),
        "decode_sample_step": int(args.decode_sample_step) if args.decode_sample_step is not None else None,
        "best_obj": {
            "mean": mean_or_none([float(r["best_obj"]) for r in per_run_records]),
            "stdev": stdev_or_zero([float(r["best_obj"]) for r in per_run_records]),
            "min": min(float(r["best_obj"]) for r in per_run_records) if per_run_records else None,
            "max": max(float(r["best_obj"]) for r in per_run_records) if per_run_records else None,
        },
        "best_T": {
            "mean": mean_or_none([float(r["best_T"]) for r in per_run_records]),
            "stdev": stdev_or_zero([float(r["best_T"]) for r in per_run_records]),
        },
        "best_C": {
            "mean": mean_or_none([float(r["best_C"]) for r in per_run_records]),
            "stdev": stdev_or_zero([float(r["best_C"]) for r in per_run_records]),
        },
        "runtime": {
            "mean": mean_or_none([float(r["runtime"]) for r in per_run_records]),
            "stdev": stdev_or_zero([float(r["runtime"]) for r in per_run_records]),
        },
        "full_decode_count": {
            "mean": mean_or_none([float(r["full_decode_count"]) for r in per_run_records]),
        },
        "generated_solution_count": {
            "mean": mean_or_none([float(r["generated_solution_count"]) for r in per_run_records]),
        },
        "llm_call_count": {
            "mean": mean_or_none([float(r["llm_call_count"]) for r in per_run_records]),
        },
        "llm_parse_fallbacks": {
            "mean": mean_or_none([float(r["llm_parse_fallbacks"]) for r in per_run_records]),
        },
        "llm_choice_a_count": {
            "mean": mean_or_none([float(r["llm_choice_a_count"]) for r in per_run_records]),
        },
        "llm_choice_b_count": {
            "mean": mean_or_none([float(r["llm_choice_b_count"]) for r in per_run_records]),
        },
        "llm_choice_a_rate": {
            "mean": mean_or_none([float(r["llm_choice_a_rate"]) for r in per_run_records]),
        },
        "llm_choice_b_rate": {
            "mean": mean_or_none([float(r["llm_choice_b_rate"]) for r in per_run_records]),
        },
        "llm_verify_cases": {
            "mean": mean_or_none([float(r["llm_verify_cases"]) for r in per_run_records]),
        },
        "llm_verify_hit_rate": {
            "mean": mean_or_none([float(r["llm_verify_hit_rate"]) for r in per_run_records]),
        },
        "llm_topk_overlap_ratio": {
            "mean": mean_or_none([float(r["llm_topk_overlap_ratio"]) for r in per_run_records]),
        },
        "llm_pairwise_hit_rate": {
            "mean": mean_or_none([float(r["llm_pairwise_hit_rate"]) for r in per_run_records]),
        },
        "scout_activation_count": {
            "mean": mean_or_none([float(r["scout_activation_count"]) for r in per_run_records]),
        },
        "scout_success_iteration_count": {
            "mean": mean_or_none([float(r["scout_success_iteration_count"]) for r in per_run_records]),
        },
        "scout_triggered_solution_count": {
            "mean": mean_or_none([float(r["scout_triggered_solution_count"]) for r in per_run_records]),
        },
        "pathrisk_repair_count": {
            "mean": mean_or_none([float(r["pathrisk_repair_count"]) for r in per_run_records]),
        },
        "repair_worth_check_count": {
            "mean": mean_or_none([float(r.get("repair_worth_check_count", 0.0)) for r in per_run_records]),
        },
        "repair_worth_accept_count": {
            "mean": mean_or_none([float(r.get("repair_worth_accept_count", 0.0)) for r in per_run_records]),
        },
        "repair_worth_reject_count": {
            "mean": mean_or_none([float(r.get("repair_worth_reject_count", 0.0)) for r in per_run_records]),
        },
        "repair_worth_accept_rate": {
            "mean": mean_ignore_none([r.get("repair_worth_accept_rate") for r in per_run_records]),
        },
        "scout_early_accept_count": {
            "mean": mean_or_none([float(r.get("scout_early_accept_count", 0.0)) for r in per_run_records]),
        },
        "scout_tardy_promote_count": {
            "mean": mean_or_none([float(r["scout_tardy_promote_count"]) for r in per_run_records]),
        },
        "scout_tardy_promote_success_count": {
            "mean": mean_or_none([float(r["scout_tardy_promote_success_count"]) for r in per_run_records]),
        },
        "successful_recovery_count": {
            "mean": mean_or_none([float(r["successful_recovery_count"]) for r in per_run_records]),
        },
        "recovery_rate": {
            "mean": mean_positive_only([r.get("recovery_rate") for r in per_run_records]),
        },
        "scout_avg_improvement": {
            "mean": mean_ignore_none([r.get("scout_avg_improvement") for r in per_run_records]),
        },
        "initial_best_Z": {
            "mean": mean_or_none([float(r["initial_best_Z"]) for r in per_run_records]),
        },
        "initial_avg_Z": {
            "mean": mean_or_none([float(r["initial_avg_Z"]) for r in per_run_records]),
        },
        "initial_std_Z": {
            "mean": mean_or_none([float(r["initial_std_Z"]) for r in per_run_records]),
        },
        "best_found_iteration": {
            "mean": mean_or_none([float(r["best_found_iteration"]) for r in per_run_records if r["best_found_iteration"] is not None]),
        },
        "initial_KBR": {
            "mean": mean_or_none([float(r["initial_KBR"]) for r in per_run_records]),
        },
        "KBR": {
            "mean": mean_or_none([float(r["KBR"]) for r in per_run_records]),
        },
        "pressure_index": {
            "mean": mean_or_none([float(r["pressure_index"]) for r in per_run_records]),
        },
        "convergence_curve": aggregate_convergence_curve(per_run_convergence_rows),
    }

    batch_summary_path = batch_dir / "batch_summary.json"
    with batch_summary_path.open("w", encoding="utf-8") as f:
        json.dump(batch_summary, f, ensure_ascii=False, indent=2)

    print(f"\nBatch manifest: {manifest_path}")
    print(f"Batch summary: {batch_summary_path}")
    print(f"Run folder: {batch_dir}")


if __name__ == "__main__":
    main()
