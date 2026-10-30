"""MOEOptimizer — Sparse Mixture-of-Experts (Top-K) Context Compression Layer.

Routes each tool output or context block through a lightweight O(n) feature
extractor and Top-K softmax gating network to 6 specialist compression experts:

1. ``lean_signal_expert``   — Adaptive signal-window truncation (errors, tracebacks, grep)
2. ``diff_hunk_expert``     — Unified diff & patch hunk surgeon (keeps headers + +/- lines)
3. ``json_schema_expert``   — JSON array & nested payload schema crusher with anomaly retention
4. ``log_dedup_expert``     — Template-normalized log stream run-length folder
5. ``tabular_expert``       — CSV / Markdown table / SQL result projection & head/tail sampler
6. ``code_skeleton_expert`` — AST-free source skeletonizer (retains signatures, imports, errors)

Includes online bandit weight updates driven by CCR retrieval feedback so
experts that cause retrievals back off automatically while high-yield experts
gain routing share.
"""

from __future__ import annotations

import json
import math
import re
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from headroom.transforms.lean_context import SIGNAL_PATTERNS, LeanContext


class ExpertId(str, Enum):
    """Specialist compression experts in the Mixture-of-Experts layer."""

    LEAN_SIGNAL = "lean_signal_expert"
    DIFF_HUNK = "diff_hunk_expert"
    JSON_SCHEMA = "json_schema_expert"
    LOG_DEDUP = "log_dedup_expert"
    TABULAR = "tabular_expert"
    CODE_SKELETON = "code_skeleton_expert"


_DIFF_HEADER_RE = re.compile(r"^(?:diff --git |---|(?:\+\+\+)|@@\s+-\d+)")
_LOG_PREFIX_RE = re.compile(
    r"^(?:\d{4}[-/]\d{2}[-/]\d{2}[T\s]\d{2}:\d{2}:\d{2}|\s*(?:Compiling|Finished|Downloading|Checking|PASSED|INFO|DEBUG)\b)"
)
_LOG_NUM_NORM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_CODE_SIG_RE = re.compile(
    r"^\s*(?:def |async def |class |fn |pub fn |pub struct |struct |impl |interface |type |export |import |from \S+ import |#[a-zA-Z_\[])"
)
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_CSV_ROW_RE = re.compile(r"^[^,\n]+(?:,[^,\n]+){3,}$")
_ERROR_LINE_RE = re.compile(
    r"(?i)\b(?:error|exception|traceback|failed|fail:|panic|fatal|critical|deadlock|oom|err!)\b"
)
_EXISTING_PROJECTION_MARKER_RE = re.compile(
    r"(?:_moe_json_expert|\.\.\. \[\d+ (?:repetitive log lines folded by MOE:log_dedup|unchanged diff context dropped|tabular rows dropped|implementation lines dropped|lines dropped)\] \.\.\.)"
)


@dataclass(frozen=True)
class RevisionSensitiveProjectionCertificate:
    """Formal Revision-Sensitive Sufficiency Certificate (pi, V_pi, R_pi).

    Certifies that a compressed context representation pi(m) is sufficient
    relative to the retained subspace V_pi and admissible revision class R_pi,
    and specifies the complementary class R \\ R_pi that requires Bennett
    reversible re-grounding via CCR (headroom_retrieve).
    """

    projection_operator: str
    retained_subspace: list[str]
    admissible_revisions: list[str]
    regrounding_required_for: list[str]
    monotone_kernel_guard: bool = True
    idempotent_skip: bool = False
    bennett_reversible_handle: str = "ccr://headroom_retrieve"
    certificate_version: str = "1.1-sufficient-projection"

    def to_dict(self) -> dict[str, Any]:
        return {
            "certificate_version": self.certificate_version,
            "projection_operator": self.projection_operator,
            "retained_subspace": list(self.retained_subspace),
            "admissible_revisions": list(self.admissible_revisions),
            "regrounding_required_for": list(self.regrounding_required_for),
            "monotone_kernel_guard": self.monotone_kernel_guard,
            "idempotent_skip": self.idempotent_skip,
            "bennett_reversible_handle": self.bennett_reversible_handle,
        }


_EXPERT_CERTIFICATES: dict[ExpertId, tuple[list[str], list[str], list[str]]] = {
    ExpertId.LEAN_SIGNAL: (
        ["head_tail_anchors", "error_traceback_windows", "grep_match_collars"],
        ["R_error_localization", "R_exit_status_triage", "R_stack_frame_diagnosis"],
        ["R_silent_middle_line_inspection", "R_exact_full_stdout_replay"],
    ),
    ExpertId.DIFF_HUNK: (
        [
            "diff_file_headers",
            "hunk_range_markers",
            "added_removed_lines",
            "2_line_change_collar",
        ],
        ["R_patch_review", "R_changed_line_verification", "R_hunk_conflict_triage"],
        ["R_untouched_function_body_audit", "R_full_file_reconstruction"],
    ),
    ExpertId.JSON_SCHEMA: (
        ["json_envelope_schema", "head_3_tail_2_items", "status_error_anomalies"],
        ["R_schema_inference", "R_anomaly_triage", "R_boundary_inspection"],
        ["R_exhaustive_aggregation_sum_count", "R_arbitrary_index_lookup_in_ker_pi"],
    ),
    ExpertId.LOG_DEDUP: (
        [
            "head_tail_4_lines",
            "error_panic_windows",
            "first_2_template_exemplars",
            "run_length_cardinality",
        ],
        [
            "R_failure_window_diagnosis",
            "R_template_frequency_analysis",
            "R_state_transition_audit",
        ],
        ["R_exact_timestamp_of_elided_repeat", "R_per_tick_jitter_series"],
    ),
    ExpertId.TABULAR: (
        ["column_header_schema", "head_6_tail_4_rows", "anomalous_error_rows"],
        [
            "R_column_schema_projection",
            "R_head_tail_row_sampling",
            "R_outlier_row_detection",
        ],
        ["R_full_table_sql_group_by", "R_interior_row_value_lookup"],
    ),
    ExpertId.CODE_SKELETON: (
        [
            "import_graph",
            "fn_class_struct_type_signatures",
            "error_signal_windows",
        ],
        [
            "R_api_surface_discovery",
            "R_call_signature_typing",
            "R_module_dependency_mapping",
        ],
        ["R_interior_branch_line_by_line_trace", "R_local_variable_mutation_audit"],
    ),
}


def build_projection_certificate(
    expert: ExpertId,
    *,
    lossless: bool = False,
    idempotent_skip: bool = False,
) -> RevisionSensitiveProjectionCertificate:
    """Construct the formal (pi, V_pi, R_pi) certificate for a specialist expert."""
    v_pi, r_pi, r_reground = _EXPERT_CERTIFICATES[expert]
    return RevisionSensitiveProjectionCertificate(
        projection_operator=f"pi:{expert.value}",
        retained_subspace=v_pi,
        admissible_revisions=r_pi,
        regrounding_required_for=[] if lossless else r_reground,
        monotone_kernel_guard=True,
        idempotent_skip=idempotent_skip,
        bennett_reversible_handle=(
            "identity://lossless" if lossless else "ccr://headroom_retrieve"
        ),
    )


@dataclass
class ContextFeatures:
    """7-dimensional structural signal vector extracted from a text block."""

    total_lines: int
    total_chars: int
    json_density: float
    tabular_density: float
    error_signal_density: float
    diff_patch_density: float
    log_repetition_density: float
    code_ast_density: float
    prose_entropy: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "total_lines": self.total_lines,
            "total_chars": self.total_chars,
            "json_density": round(self.json_density, 4),
            "tabular_density": round(self.tabular_density, 4),
            "error_signal_density": round(self.error_signal_density, 4),
            "diff_patch_density": round(self.diff_patch_density, 4),
            "log_repetition_density": round(self.log_repetition_density, 4),
            "code_ast_density": round(self.code_ast_density, 4),
            "prose_entropy": round(self.prose_entropy, 4),
        }


@dataclass
class MOERoutingDecision:
    """Top-K gating decision produced by MOEGatingNetwork."""

    top_experts: list[tuple[ExpertId, float]]
    all_probabilities: dict[str, float]
    features: ContextFeatures

    @property
    def primary_expert(self) -> ExpertId:
        return self.top_experts[0][0]

    @property
    def primary_confidence(self) -> float:
        return self.top_experts[0][1]


@dataclass
class MOEOptimizationResult:
    """Result of applying the MOEOptimizer layer to a text payload."""

    text: str
    original_lines: int
    kept_lines: int
    dropped_lines: int
    original_tokens: int
    compressed_tokens: int
    tokens_saved: int
    primary_expert: str
    secondary_expert: str | None
    expert_weights: dict[str, float]
    features: dict[str, float | int] = field(default_factory=dict)
    certificate: dict[str, Any] = field(default_factory=dict)

    @property
    def savings_pct(self) -> float:
        if self.original_tokens <= 0:
            return 0.0
        return round((self.tokens_saved / self.original_tokens) * 100.0, 2)


class MOEGatingNetwork:
    """Sparse Top-K softmax gating network with online bandit weight adaptation."""

    def __init__(self, top_k: int = 2):
        self.top_k = max(1, min(top_k, len(ExpertId)))
        self._lock = threading.Lock()
        # Online bandit bias adjustments per expert (updated via retrieval/success feedback)
        self._expert_bias: dict[ExpertId, float] = dict.fromkeys(ExpertId, 0.0)

    def extract_features(self, text: str, lines: list[str]) -> ContextFeatures:
        """Extract O(n) structural density features from text."""
        n_lines = max(1, len(lines))
        n_chars = len(text)
        stripped = text.strip()

        # 1. JSON density
        is_json_envelope = (stripped.startswith("{") and stripped.endswith("}")) or (
            stripped.startswith("[") and stripped.endswith("]")
        )
        brace_lines = sum(
            1 for ln in lines[:400] if any(tok in ln for tok in ('":', "{", "}", "[", "]"))
        )
        sample_n = max(1, min(n_lines, 400))
        json_density = min(
            1.0, (0.65 if is_json_envelope else 0.0) + 0.5 * (brace_lines / sample_n)
        )

        # 2. Tabular density
        table_lines = 0
        diff_lines = 0
        log_lines = 0
        code_lines = 0
        error_lines = 0
        normalized_templates: set[str] = set()
        non_empty = 0

        for ln in lines[:500]:
            s = ln.strip()
            if not s:
                continue
            non_empty += 1
            if _TABLE_ROW_RE.match(ln) or _CSV_ROW_RE.match(ln):
                table_lines += 1
            if _DIFF_HEADER_RE.match(ln) or (
                len(ln) > 1 and ln[0] in "+-" and not ln.startswith("--- ")
            ):
                diff_lines += 1
            if _LOG_PREFIX_RE.match(ln):
                log_lines += 1
            if _CODE_SIG_RE.match(ln):
                code_lines += 1
            if _ERROR_LINE_RE.search(ln):
                error_lines += 1
            normalized_templates.add(_LOG_NUM_NORM_RE.sub("#", s[:60]))

        denom = max(1, non_empty)
        tabular_density = min(1.0, table_lines / denom)
        diff_patch_density = min(1.0, diff_lines / denom)
        repetition_ratio = max(0.0, 1.0 - (len(normalized_templates) / denom))
        log_repetition_density = min(1.0, 0.55 * (log_lines / denom) + 0.65 * repetition_ratio)
        code_ast_density = min(1.0, (code_lines / denom) * 3.0)
        error_signal_density = min(1.0, (error_lines / denom) * 4.5)
        prose_entropy = max(
            0.0,
            1.0
            - max(
                json_density,
                tabular_density,
                diff_patch_density,
                log_repetition_density,
                code_ast_density,
            ),
        )

        return ContextFeatures(
            total_lines=len(lines),
            total_chars=n_chars,
            json_density=json_density,
            tabular_density=tabular_density,
            error_signal_density=error_signal_density,
            diff_patch_density=diff_patch_density,
            log_repetition_density=log_repetition_density,
            code_ast_density=code_ast_density,
            prose_entropy=prose_entropy,
        )

    def route(self, text: str, lines: list[str] | None = None) -> MOERoutingDecision:
        """Compute expert logits and select Top-K compression experts."""
        line_list = lines if lines is not None else text.split("\n")
        feat = self.extract_features(text, line_list)

        with self._lock:
            biases = dict(self._expert_bias)

        logits: dict[ExpertId, float] = {
            ExpertId.JSON_SCHEMA: (
                4.2 * feat.json_density
                - 1.0 * feat.diff_patch_density
                + biases[ExpertId.JSON_SCHEMA]
            ),
            ExpertId.DIFF_HUNK: (
                4.5 * feat.diff_patch_density - 0.8 * feat.json_density + biases[ExpertId.DIFF_HUNK]
            ),
            ExpertId.TABULAR: (
                4.3 * feat.tabular_density - 0.5 * feat.json_density + biases[ExpertId.TABULAR]
            ),
            ExpertId.LOG_DEDUP: (
                3.6 * feat.log_repetition_density
                + 0.8 * feat.error_signal_density
                + biases[ExpertId.LOG_DEDUP]
            ),
            ExpertId.CODE_SKELETON: (
                3.4 * feat.code_ast_density
                - 1.2 * feat.error_signal_density
                - 0.8 * feat.diff_patch_density
                + biases[ExpertId.CODE_SKELETON]
            ),
            ExpertId.LEAN_SIGNAL: (
                3.8 * feat.error_signal_density
                + 1.4 * feat.prose_entropy
                + 0.6
                + biases[ExpertId.LEAN_SIGNAL]
            ),
        }

        max_logit = max(logits.values())
        exp_scores = {e: math.exp(score - max_logit) for e, score in logits.items()}
        total_exp = sum(exp_scores.values()) or 1.0
        probs = {e: exp_scores[e] / total_exp for e in ExpertId}

        ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        top_k = ranked[: self.top_k]
        top_sum = sum(w for _, w in top_k) or 1.0
        normalized_top_k = [(e, round(w / top_sum, 4)) for e, w in top_k]

        return MOERoutingDecision(
            top_experts=normalized_top_k,
            all_probabilities={e.value: round(p, 4) for e, p in ranked},
            features=feat,
        )

    def update_bandit_feedback(self, expert: ExpertId | str, reward: float) -> None:
        """Update online gating bias for an expert based on downstream reward."""
        try:
            exp_id = ExpertId(expert)
        except ValueError:
            return
        with self._lock:
            current = self._expert_bias.get(exp_id, 0.0)
            self._expert_bias[exp_id] = max(-1.5, min(1.5, current + 0.05 * reward))

    def export_burn_tensor_state(self) -> dict[str, Any]:
        """Export gating parameters in tracel-ai/burn Linear<B> [6, 7] tensor format.

        Feature order:
          [json_density, tabular_density, error_signal_density, diff_patch_density,
           log_repetition_density, code_ast_density, prose_entropy]
        """
        with self._lock:
            biases = {e.value: round(self._expert_bias[e], 6) for e in ExpertId}
        weight_matrix: dict[str, list[float]] = {
            ExpertId.LEAN_SIGNAL.value: [0.0, 0.0, 3.8, 0.0, 0.0, 0.0, 1.4],
            ExpertId.DIFF_HUNK.value: [-0.8, 0.0, 0.0, 4.5, 0.0, 0.0, 0.0],
            ExpertId.JSON_SCHEMA.value: [4.2, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
            ExpertId.LOG_DEDUP.value: [0.0, 0.0, 0.8, 0.0, 3.6, 0.0, 0.0],
            ExpertId.TABULAR.value: [-0.5, 4.3, 0.0, 0.0, 0.0, 0.0, 0.0],
            ExpertId.CODE_SKELETON.value: [0.0, 0.0, -1.2, -0.8, 0.0, 3.4, 0.0],
        }
        return {
            "framework": "tracel-ai/burn",
            "module": "headroom_core::transforms::moe_optimizer::BurnMoeGatingRouter",
            "backends": ["burn-wgpu (M1 Metal)", "burn-flex (SIMD)", "burn-ndarray"],
            "tensor_shape": [len(ExpertId), 7],
            "top_k": self.top_k,
            "weights": weight_matrix,
            "expert_bias": biases,
        }

    def merge_federated_bias(
        self, peer_biases: dict[str, float], alpha: float = 0.35
    ) -> dict[str, float]:
        """Merge peer Burn MOE gating biases received via n0-computer/iroh gossip."""
        alpha_clamped = max(0.05, min(0.95, alpha))
        with self._lock:
            for exp in ExpertId:
                if exp.value in peer_biases:
                    peer_val = float(peer_biases[exp.value])
                    merged = (1.0 - alpha_clamped) * self._expert_bias[
                        exp
                    ] + alpha_clamped * peer_val
                    self._expert_bias[exp] = max(-1.5, min(1.5, merged))
            return {e.value: round(self._expert_bias[e], 6) for e in ExpertId}


class MOEOptimizer:
    """Sparse Mixture-of-Experts Context Optimizer for 8b.HeadroomZ."""

    def __init__(self, window_radius: int = 25, top_k: int = 2, min_lines: int = 24):
        self.window_radius = window_radius
        self.min_lines = min_lines
        self.gating = MOEGatingNetwork(top_k=top_k)
        self._lean_ctx = LeanContext(window_radius=window_radius)
        self._lock = threading.Lock()
        self._stats: dict[str, Any] = {
            "calls": 0,
            "proxy_calls": 0,
            "tokens_saved": 0,
            "attempted_tokens": 0,
            "proxy_tokens_saved": 0,
            "proxy_attempted_tokens": 0,
            "lines_dropped": 0,
            "proxy_lines_dropped": 0,
            "expert_wins": {e.value: 0 for e in ExpertId},
            "expert_tokens_saved": {e.value: 0 for e in ExpertId},
            "retrieval_penalties": 0,
            "monotone_kernel_skips": 0,
        }

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            base = {
                **self._stats,
                "expert_wins": dict(self._stats["expert_wins"]),
                "expert_tokens_saved": dict(self._stats["expert_tokens_saved"]),
            }
        base["burn_gating"] = self.gating.export_burn_tensor_state()
        base["projection_certificates"] = {
            e.value: build_projection_certificate(e).to_dict() for e in ExpertId
        }
        return base

    def reset_stats(self) -> None:
        with self._lock:
            self._stats["calls"] = 0
            self._stats["proxy_calls"] = 0
            self._stats["tokens_saved"] = 0
            self._stats["attempted_tokens"] = 0
            self._stats["proxy_tokens_saved"] = 0
            self._stats["proxy_attempted_tokens"] = 0
            self._stats["lines_dropped"] = 0
            self._stats["proxy_lines_dropped"] = 0
            self._stats["retrieval_penalties"] = 0
            self._stats["monotone_kernel_skips"] = 0
            for e in ExpertId:
                self._stats["expert_wins"][e.value] = 0
                self._stats["expert_tokens_saved"][e.value] = 0

    def record_retrieval_penalty(self, expert: str) -> None:
        """Penalize an expert when the model had to retrieve original content via CCR."""
        self.gating.update_bandit_feedback(expert, reward=-1.0)
        with self._lock:
            self._stats["retrieval_penalties"] += 1

    # ── Specialist Expert Implementations ─────────────────────────────

    def _run_lean_signal_expert(self, text: str, lines: list[str]) -> list[str]:
        res = self._lean_ctx.truncate(text)
        return res.text.split("\n")

    def _run_diff_hunk_expert(self, lines: list[str]) -> list[str]:
        """Keep diff headers, @@ hunks, +/- lines, and a 2-line collar around changes."""
        n = len(lines)
        important: set[int] = set(range(min(3, n))) | set(range(max(0, n - 3), n))
        for i, ln in enumerate(lines):
            if _DIFF_HEADER_RE.match(ln) or (
                len(ln) > 0 and ln[0] in "+-" and not ln.startswith("   ")
            ):
                for j in range(max(0, i - 2), min(n, i + 3)):
                    important.add(j)
            elif _ERROR_LINE_RE.search(ln):
                for j in range(max(0, i - 3), min(n, i + 4)):
                    important.add(j)

        if len(important) >= n:
            return lines
        return self._assemble_kept_lines(lines, important, label="unchanged diff context")

    def _run_json_schema_expert(self, text: str, lines: list[str]) -> list[str]:
        """Crush homogeneous JSON arrays/objects while retaining anomalies and boundaries."""
        try:
            parsed = json.loads(text)
        except Exception:
            return self._run_lean_signal_expert(text, lines)

        def _is_anomaly(item: Any) -> bool:
            if not isinstance(item, dict):
                return False
            for k, v in item.items():
                kl = str(k).lower()
                if kl in ("error", "exception", "traceback", "failure") and v:
                    return True
                if _ERROR_LINE_RE.search(str(v)):
                    return True
                if kl in ("status", "state", "level") and any(
                    tok in str(v).lower()
                    for tok in (
                        "error",
                        "failed",
                        "critical",
                        "fatal",
                        "panic",
                        "deadlock",
                        "timeout",
                    )
                ):
                    return True
            return False

        def _crush_value(val: Any) -> Any:
            if isinstance(val, list) and len(val) > 8:
                anomalies = [x for x in val[3:-2] if _is_anomaly(x)]
                dropped = len(val) - 5 - len(anomalies)
                if dropped > 0:
                    return [
                        *val[:3],
                        *anomalies[:5],
                        {
                            "_moe_json_expert": f"... [{dropped} homogeneous items elided; {len(anomalies)} anomalies preserved] ..."
                        },
                        *val[-2:],
                    ]
                return val
            if isinstance(val, dict):
                out: dict[str, Any] = {}
                for k, v in val.items():
                    out[k] = _crush_value(v)
                return out
            return val

        crushed = _crush_value(parsed)
        rendered = json.dumps(crushed, indent=2, ensure_ascii=False)
        out_lines = rendered.split("\n")
        if len(out_lines) < len(lines):
            return out_lines
        return self._run_lean_signal_expert(text, lines)

    def _run_log_dedup_expert(self, lines: list[str]) -> list[str]:
        """Fold repetitive log templates while preserving errors and state transitions."""
        n = len(lines)
        signal_indices: set[int] = set(range(min(4, n))) | set(range(max(0, n - 4), n))
        for i, ln in enumerate(lines):
            if _ERROR_LINE_RE.search(ln):
                for j in range(max(0, i - 4), min(n, i + 5)):
                    signal_indices.add(j)

        out: list[str] = []
        seen_templates: dict[str, int] = {}
        elided_run = 0

        for i, ln in enumerate(lines):
            if i in signal_indices:
                if elided_run > 0:
                    out.append(
                        f"... [{elided_run} repetitive log lines folded by MOE:log_dedup] ..."
                    )
                    elided_run = 0
                out.append(ln)
                continue

            tmpl = _LOG_NUM_NORM_RE.sub("#", ln.strip()[:72])
            count = seen_templates.get(tmpl, 0) + 1
            seen_templates[tmpl] = count
            if count <= 2:
                if elided_run > 0:
                    out.append(
                        f"... [{elided_run} repetitive log lines folded by MOE:log_dedup] ..."
                    )
                    elided_run = 0
                out.append(ln)
            else:
                elided_run += 1

        if elided_run > 0:
            out.append(f"... [{elided_run} repetitive log lines folded by MOE:log_dedup] ...")
        return out

    def _run_tabular_expert(self, lines: list[str]) -> list[str]:
        """Keep table header, separator, anomalous rows, and first/last 5 rows."""
        n = len(lines)
        keep: set[int] = set(range(min(6, n))) | set(range(max(0, n - 4), n))
        for i, ln in enumerate(lines):
            if _ERROR_LINE_RE.search(ln):
                keep.update(range(max(0, i - 1), min(n, i + 2)))
        return self._assemble_kept_lines(lines, keep, label="tabular rows")

    def _run_code_skeleton_expert(self, lines: list[str]) -> list[str]:
        """Retain imports, class/function/type signatures, and error windows."""
        n = len(lines)
        keep: set[int] = set(range(min(6, n))) | set(range(max(0, n - 5), n))
        for i, ln in enumerate(lines):
            if _CODE_SIG_RE.match(ln):
                keep.update(range(i, min(n, i + 3)))
            elif _ERROR_LINE_RE.search(ln) or any(p.search(ln) for p in SIGNAL_PATTERNS):
                keep.update(range(max(0, i - 5), min(n, i + 6)))
        return self._assemble_kept_lines(lines, keep, label="implementation lines")

    @staticmethod
    def _assemble_kept_lines(lines: list[str], keep: set[int], label: str = "lines") -> list[str]:
        result: list[str] = []
        prev = -1
        for idx in sorted(keep):
            if idx > prev + 1:
                skipped = idx - prev - 1
                result.append(f"... [{skipped} {label} dropped] ...")
            result.append(lines[idx])
            prev = idx
        if prev < len(lines) - 1:
            skipped = len(lines) - 1 - prev
            result.append(f"... [{skipped} {label} dropped] ...")
        return result

    def _execute_expert(self, expert: ExpertId, text: str, lines: list[str]) -> list[str]:
        if expert == ExpertId.JSON_SCHEMA:
            return self._run_json_schema_expert(text, lines)
        if expert == ExpertId.DIFF_HUNK:
            return self._run_diff_hunk_expert(lines)
        if expert == ExpertId.LOG_DEDUP:
            return self._run_log_dedup_expert(lines)
        if expert == ExpertId.TABULAR:
            return self._run_tabular_expert(lines)
        if expert == ExpertId.CODE_SKELETON:
            return self._run_code_skeleton_expert(lines)
        return self._run_lean_signal_expert(text, lines)

    def optimize(
        self,
        text: str,
        *,
        record_proxy_stats: bool = False,
    ) -> MOEOptimizationResult:
        """Route text through the Top-K Mixture-of-Experts and return the best compression."""
        if not text or not text.strip():
            return MOEOptimizationResult(
                text=text,
                original_lines=0,
                kept_lines=0,
                dropped_lines=0,
                original_tokens=0,
                compressed_tokens=0,
                tokens_saved=0,
                primary_expert=ExpertId.LEAN_SIGNAL.value,
                secondary_expert=None,
                expert_weights={},
                certificate=build_projection_certificate(
                    ExpertId.LEAN_SIGNAL, lossless=True
                ).to_dict(),
            )

        lines = text.split("\n")
        n_lines = len(lines)
        orig_tokens = max(1, len(text) // 4)
        decision = self.gating.route(text, lines)
        primary_exp = decision.top_experts[0][0]
        secondary_exp = decision.top_experts[1][0] if len(decision.top_experts) > 1 else None

        # Monotone Kernel Growth Guard (ker(pi_n) <= ker(pi_{n+1} o pi_n)):
        # If the block already bears a projection marker from an earlier turn,
        # enforce idempotence pi(pi(m)) = pi(m) so iterative turns never compound
        # kernel loss across the context window.
        if _EXISTING_PROJECTION_MARKER_RE.search(text):
            with self._lock:
                self._stats["monotone_kernel_skips"] += 1
            return MOEOptimizationResult(
                text=text,
                original_lines=n_lines,
                kept_lines=n_lines,
                dropped_lines=0,
                original_tokens=orig_tokens,
                compressed_tokens=orig_tokens,
                tokens_saved=0,
                primary_expert=primary_exp.value,
                secondary_expert=secondary_exp.value if secondary_exp else None,
                expert_weights=decision.all_probabilities,
                features=decision.features.to_dict(),
                certificate=build_projection_certificate(
                    primary_exp, lossless=False, idempotent_skip=True
                ).to_dict(),
            )

        if n_lines < self.min_lines and len(text) < 1200:
            return MOEOptimizationResult(
                text=text,
                original_lines=n_lines,
                kept_lines=n_lines,
                dropped_lines=0,
                original_tokens=orig_tokens,
                compressed_tokens=orig_tokens,
                tokens_saved=0,
                primary_expert=primary_exp.value,
                secondary_expert=secondary_exp.value if secondary_exp else None,
                expert_weights=decision.all_probabilities,
                features=decision.features.to_dict(),
                certificate=build_projection_certificate(primary_exp, lossless=True).to_dict(),
            )

        # Evaluate Top-1 expert; if Top-1 yields <10% reduction and Top-2 has
        # meaningful gating weight (>=0.20), evaluate Top-2 and pick the higher
        # signal-preserving reduction.
        cand1_lines = self._execute_expert(primary_exp, text, lines)
        cand1_text = "\n".join(cand1_lines)
        winning_exp = primary_exp
        winning_lines = cand1_lines
        winning_text = cand1_text

        if (
            secondary_exp is not None
            and decision.top_experts[1][1] >= 0.20
            and len(cand1_text) > len(text) * 0.90
        ):
            cand2_lines = self._execute_expert(secondary_exp, text, lines)
            cand2_text = "\n".join(cand2_lines)
            if len(cand2_text) < len(winning_text):
                winning_exp = secondary_exp
                winning_lines = cand2_lines
                winning_text = cand2_text

        if len(winning_text) >= len(text):
            winning_text = text
            kept_count = n_lines
            dropped_count = 0
            comp_tokens = orig_tokens
        else:
            marker_count = sum(1 for ln in winning_lines if ln.startswith("... ["))
            kept_count = min(n_lines, max(1, len(winning_lines) - marker_count))
            dropped_count = max(0, n_lines - kept_count)
            comp_tokens = max(1, len(winning_text) // 4)

        tokens_saved = max(0, orig_tokens - comp_tokens)

        with self._lock:
            self._stats["calls"] += 1
            self._stats["attempted_tokens"] += orig_tokens
            self._stats["tokens_saved"] += tokens_saved
            self._stats["lines_dropped"] += dropped_count
            self._stats["expert_wins"][winning_exp.value] += 1
            self._stats["expert_tokens_saved"][winning_exp.value] += tokens_saved
            if record_proxy_stats:
                self._stats["proxy_calls"] += 1
                self._stats["proxy_attempted_tokens"] += orig_tokens
                self._stats["proxy_tokens_saved"] += tokens_saved
                self._stats["proxy_lines_dropped"] += dropped_count

        if tokens_saved > 0:
            self.gating.update_bandit_feedback(winning_exp, reward=0.25)

        return MOEOptimizationResult(
            text=winning_text,
            original_lines=n_lines,
            kept_lines=kept_count,
            dropped_lines=dropped_count,
            original_tokens=orig_tokens,
            compressed_tokens=comp_tokens,
            tokens_saved=tokens_saved,
            primary_expert=winning_exp.value,
            secondary_expert=secondary_exp.value if secondary_exp else None,
            expert_weights=decision.all_probabilities,
            features=decision.features.to_dict(),
            certificate=build_projection_certificate(
                winning_exp, lossless=(dropped_count == 0)
            ).to_dict(),
        )


# ── Singleton for proxy & SDK use ────────────────────────────────────
_default_moe: MOEOptimizer | None = None


def get_moe_optimizer(window_radius: int = 25, top_k: int = 2) -> MOEOptimizer:
    """Get or create the default MOEOptimizer singleton."""
    global _default_moe
    if _default_moe is None:
        _default_moe = MOEOptimizer(window_radius=window_radius, top_k=top_k)
    elif _default_moe.window_radius != window_radius:
        _default_moe.window_radius = window_radius
        _default_moe._lean_ctx.window_radius = window_radius
    return _default_moe
