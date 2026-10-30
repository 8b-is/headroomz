//! Burn-compatible Sparse Mixture-of-Experts (`MoeOptimizer`) context router.
//!
//! Implements the `tracel-ai/burn` Linear<B> `[6, 7]` gating projection and
//! Top-K softmax router in pure Rust so `headroom-core` and `headroom-proxy`
//! can route tool outputs to the 6 specialist compression experts with zero
//! heap contention and export/import Burn tensor checkpoints federated over
//! `n0-computer/iroh`.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Specialist compression experts in the 8b.HeadroomZ Mixture-of-Experts layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MoeExpertId {
    LeanSignalExpert,
    DiffHunkExpert,
    JsonSchemaExpert,
    LogDedupExpert,
    TabularExpert,
    CodeSkeletonExpert,
}

impl MoeExpertId {
    pub const ALL: [MoeExpertId; 6] = [
        MoeExpertId::LeanSignalExpert,
        MoeExpertId::DiffHunkExpert,
        MoeExpertId::JsonSchemaExpert,
        MoeExpertId::LogDedupExpert,
        MoeExpertId::TabularExpert,
        MoeExpertId::CodeSkeletonExpert,
    ];

    pub fn as_str(&self) -> &'static str {
        match self {
            MoeExpertId::LeanSignalExpert => "lean_signal_expert",
            MoeExpertId::DiffHunkExpert => "diff_hunk_expert",
            MoeExpertId::JsonSchemaExpert => "json_schema_expert",
            MoeExpertId::LogDedupExpert => "log_dedup_expert",
            MoeExpertId::TabularExpert => "tabular_expert",
            MoeExpertId::CodeSkeletonExpert => "code_skeleton_expert",
        }
    }

    /// Emit the formal Revision-Sensitive Projection Certificate `(\pi, V_\pi, R_\pi)`
    /// for this specialist expert.
    pub fn projection_certificate(&self, lossless: bool) -> RevisionSensitiveProjectionCertificate {
        let (retained_subspace, admissible_revisions, regrounding_required_for) = match self {
            MoeExpertId::LeanSignalExpert => (
                vec![
                    "head_tail_anchors".to_string(),
                    "error_traceback_windows".to_string(),
                    "grep_match_collars".to_string(),
                ],
                vec![
                    "R_error_localization".to_string(),
                    "R_exit_status_triage".to_string(),
                    "R_stack_frame_diagnosis".to_string(),
                ],
                vec![
                    "R_silent_middle_line_inspection".to_string(),
                    "R_exact_full_stdout_replay".to_string(),
                ],
            ),
            MoeExpertId::DiffHunkExpert => (
                vec![
                    "diff_file_headers".to_string(),
                    "hunk_range_markers".to_string(),
                    "added_removed_lines".to_string(),
                    "2_line_change_collar".to_string(),
                ],
                vec![
                    "R_patch_review".to_string(),
                    "R_changed_line_verification".to_string(),
                    "R_hunk_conflict_triage".to_string(),
                ],
                vec![
                    "R_untouched_function_body_audit".to_string(),
                    "R_full_file_reconstruction".to_string(),
                ],
            ),
            MoeExpertId::JsonSchemaExpert => (
                vec![
                    "json_envelope_schema".to_string(),
                    "head_3_tail_2_items".to_string(),
                    "status_error_anomalies".to_string(),
                ],
                vec![
                    "R_schema_inference".to_string(),
                    "R_anomaly_triage".to_string(),
                    "R_boundary_inspection".to_string(),
                ],
                vec![
                    "R_exhaustive_aggregation_sum_count".to_string(),
                    "R_arbitrary_index_lookup_in_ker_pi".to_string(),
                ],
            ),
            MoeExpertId::LogDedupExpert => (
                vec![
                    "head_tail_4_lines".to_string(),
                    "error_panic_windows".to_string(),
                    "first_2_template_exemplars".to_string(),
                    "run_length_cardinality".to_string(),
                ],
                vec![
                    "R_failure_window_diagnosis".to_string(),
                    "R_template_frequency_analysis".to_string(),
                    "R_state_transition_audit".to_string(),
                ],
                vec![
                    "R_exact_timestamp_of_elided_repeat".to_string(),
                    "R_per_tick_jitter_series".to_string(),
                ],
            ),
            MoeExpertId::TabularExpert => (
                vec![
                    "column_header_schema".to_string(),
                    "head_6_tail_4_rows".to_string(),
                    "anomalous_error_rows".to_string(),
                ],
                vec![
                    "R_column_schema_projection".to_string(),
                    "R_head_tail_row_sampling".to_string(),
                    "R_outlier_row_detection".to_string(),
                ],
                vec![
                    "R_full_table_sql_group_by".to_string(),
                    "R_interior_row_value_lookup".to_string(),
                ],
            ),
            MoeExpertId::CodeSkeletonExpert => (
                vec![
                    "import_graph".to_string(),
                    "fn_class_struct_type_signatures".to_string(),
                    "error_signal_windows".to_string(),
                ],
                vec![
                    "R_api_surface_discovery".to_string(),
                    "R_call_signature_typing".to_string(),
                    "R_module_dependency_mapping".to_string(),
                ],
                vec![
                    "R_interior_branch_line_by_line_trace".to_string(),
                    "R_local_variable_mutation_audit".to_string(),
                ],
            ),
        };
        RevisionSensitiveProjectionCertificate {
            certificate_version: "1.1-sufficient-projection".to_string(),
            projection_operator: format!("pi:{}", self.as_str()),
            retained_subspace,
            admissible_revisions,
            regrounding_required_for: if lossless {
                Vec::new()
            } else {
                regrounding_required_for
            },
            monotone_kernel_guard: true,
            bennett_reversible_handle: if lossless {
                "identity://lossless".to_string()
            } else {
                "ccr://headroom_retrieve".to_string()
            },
        }
    }
}

/// Formal Revision-Sensitive Sufficiency Certificate `(\pi, V_\pi, R_\pi)`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RevisionSensitiveProjectionCertificate {
    pub certificate_version: String,
    pub projection_operator: String,
    pub retained_subspace: Vec<String>,
    pub admissible_revisions: Vec<String>,
    pub regrounding_required_for: Vec<String>,
    pub monotone_kernel_guard: bool,
    pub bennett_reversible_handle: String,
}

/// 7-dimensional structural signal tensor `[1, 7]` extracted from a context block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MoeFeatureTensor {
    pub total_lines: usize,
    pub total_chars: usize,
    pub json_density: f32,
    pub tabular_density: f32,
    pub error_signal_density: f32,
    pub diff_patch_density: f32,
    pub log_repetition_density: f32,
    pub code_ast_density: f32,
    pub prose_entropy: f32,
}

impl MoeFeatureTensor {
    /// Convert to a 7-element `f32` vector matching the Burn `Linear<B>` input dimension.
    pub fn as_array(&self) -> [f32; 7] {
        [
            self.json_density,
            self.tabular_density,
            self.error_signal_density,
            self.diff_patch_density,
            self.log_repetition_density,
            self.code_ast_density,
            self.prose_entropy,
        ]
    }
}

/// Serializable `tracel-ai/burn` `Linear<B>` `[6, 7]` gating state dict.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BurnMoeTensorState {
    pub framework: String,
    pub tensor_shape: [usize; 2],
    pub top_k: usize,
    pub weights: [[f32; 7]; 6],
    pub bias: [f32; 6],
}

/// Top-K routing result emitted by [`BurnMoeGatingRouter`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MoeRoutingDecision {
    pub primary_expert: MoeExpertId,
    pub secondary_expert: Option<MoeExpertId>,
    pub probabilities: BTreeMap<String, f32>,
    pub features: MoeFeatureTensor,
    pub certificate: RevisionSensitiveProjectionCertificate,
}

/// `tracel-ai/burn`-aligned Top-K softmax gating router (`[6, 7]` linear projection + online bandit bias).
#[derive(Debug, Clone)]
pub struct BurnMoeGatingRouter {
    top_k: usize,
    weights: [[f32; 7]; 6],
    bias: [f32; 6],
}

impl Default for BurnMoeGatingRouter {
    fn default() -> Self {
        Self::new(2)
    }
}

impl BurnMoeGatingRouter {
    pub fn new(top_k: usize) -> Self {
        // Feature order:
        // [json, tabular, error_signal, diff_patch, log_repetition, code_ast, prose_entropy]
        let weights: [[f32; 7]; 6] = [
            // LeanSignalExpert
            [0.0, 0.0, 3.8, 0.0, 0.0, 0.0, 1.4],
            // DiffHunkExpert
            [-0.8, 0.0, 0.0, 4.5, 0.0, 0.0, 0.0],
            // JsonSchemaExpert
            [4.2, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
            // LogDedupExpert
            [0.0, 0.0, 0.8, 0.0, 3.6, 0.0, 0.0],
            // TabularExpert
            [-0.5, 4.3, 0.0, 0.0, 0.0, 0.0, 0.0],
            // CodeSkeletonExpert
            [0.0, 0.0, -1.2, -0.8, 0.0, 3.4, 0.0],
        ];
        let bias: [f32; 6] = [0.6, 0.0, 0.0, 0.0, 0.0, 0.0];
        Self {
            top_k: top_k.clamp(1, 6),
            weights,
            bias,
        }
    }

    /// Extract 7-dimensional structural signal features in O(n).
    pub fn extract_features(&self, text: &str) -> MoeFeatureTensor {
        let lines: Vec<&str> = text.lines().collect();
        let n_lines = lines.len().max(1);
        let stripped = text.trim();
        let is_json_env = (stripped.starts_with('{') && stripped.ends_with('}'))
            || (stripped.starts_with('[') && stripped.ends_with(']'));

        let sample_n = n_lines.min(400).max(1);
        let mut brace_lines = 0usize;
        let mut table_lines = 0usize;
        let mut diff_lines = 0usize;
        let mut log_lines = 0usize;
        let mut code_lines = 0usize;
        let mut error_lines = 0usize;
        let mut non_empty = 0usize;

        for ln in lines.iter().take(400) {
            let s = ln.trim();
            if s.is_empty() {
                continue;
            }
            non_empty += 1;
            if s.contains("\":") || s.contains('{') || s.contains('}') || s.contains('[') {
                brace_lines += 1;
            }
            if (s.starts_with('|') && s.ends_with('|')) || s.matches(',').count() >= 3 {
                table_lines += 1;
            }
            if s.starts_with("diff --git ")
                || s.starts_with("@@ ")
                || s.starts_with("+++ ")
                || s.starts_with("--- ")
                || (ln.len() > 1 && (ln.starts_with('+') || ln.starts_with('-')))
            {
                diff_lines += 1;
            }
            if s.starts_with("202")
                || s.starts_with("INFO")
                || s.starts_with("DEBUG")
                || s.starts_with("Compiling ")
                || s.starts_with("Checking ")
            {
                log_lines += 1;
            }
            if s.starts_with("def ")
                || s.starts_with("async def ")
                || s.starts_with("class ")
                || s.starts_with("fn ")
                || s.starts_with("pub fn ")
                || s.starts_with("pub struct ")
                || s.starts_with("impl ")
            {
                code_lines += 1;
            }
            let lower = s.to_ascii_lowercase();
            if lower.contains("error")
                || lower.contains("exception")
                || lower.contains("traceback")
                || lower.contains("panic")
                || lower.contains("failed")
            {
                error_lines += 1;
            }
        }

        let denom = (non_empty.max(1)) as f32;
        let json_density = ((if is_json_env { 0.65 } else { 0.0 })
            + 0.5 * (brace_lines as f32 / sample_n as f32))
            .min(1.0);
        let tabular_density = (table_lines as f32 / denom).min(1.0);
        let diff_patch_density = (diff_lines as f32 / denom).min(1.0);
        let log_repetition_density = (log_lines as f32 / denom).min(1.0);
        let code_ast_density = ((code_lines as f32 / denom) * 3.0).min(1.0);
        let error_signal_density = ((error_lines as f32 / denom) * 4.5).min(1.0);
        let max_struct = json_density
            .max(tabular_density)
            .max(diff_patch_density)
            .max(log_repetition_density)
            .max(code_ast_density);
        let prose_entropy = (1.0 - max_struct).max(0.0);

        MoeFeatureTensor {
            total_lines: lines.len(),
            total_chars: text.len(),
            json_density,
            tabular_density,
            error_signal_density,
            diff_patch_density,
            log_repetition_density,
            code_ast_density,
            prose_entropy,
        }
    }

    /// Forward pass `y = softmax(W_g * x + b_g)` and Top-K selection.
    pub fn route(&self, text: &str) -> MoeRoutingDecision {
        let feat = self.extract_features(text);
        let x = feat.as_array();
        let mut logits = [0.0f32; 6];
        let mut max_logit = f32::NEG_INFINITY;

        for (i, row) in self.weights.iter().enumerate() {
            let mut sum = self.bias[i];
            for (w, xi) in row.iter().zip(x.iter()) {
                sum += w * xi;
            }
            logits[i] = sum;
            if sum > max_logit {
                max_logit = sum;
            }
        }

        let mut exp_scores = [0.0f32; 6];
        let mut total_exp = 0.0f32;
        for i in 0..6 {
            let e = (logits[i] - max_logit).exp();
            exp_scores[i] = e;
            total_exp += e;
        }
        let denom = if total_exp > 0.0 { total_exp } else { 1.0 };

        let mut indexed: Vec<(MoeExpertId, f32)> = MoeExpertId::ALL
            .iter()
            .copied()
            .enumerate()
            .map(|(i, id)| (id, exp_scores[i] / denom))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut probabilities = BTreeMap::new();
        for (id, p) in &indexed {
            probabilities.insert(id.as_str().to_string(), (*p * 10000.0).round() / 10000.0);
        }

        let primary_expert = indexed[0].0;
        let secondary_expert = if self.top_k > 1 {
            Some(indexed[1].0)
        } else {
            None
        };

        let certificate = primary_expert.projection_certificate(false);

        MoeRoutingDecision {
            primary_expert,
            secondary_expert,
            probabilities,
            features: feat,
            certificate,
        }
    }

    /// Online bandit reward update (`reward < 0` on CCR retrieval penalty).
    pub fn update_bandit_feedback(&mut self, expert: MoeExpertId, reward: f32) {
        let idx = MoeExpertId::ALL
            .iter()
            .position(|e| *e == expert)
            .unwrap_or(0);
        self.bias[idx] = (self.bias[idx] + 0.05 * reward).clamp(-1.5, 1.5);
    }

    /// Export state in `tracel-ai/burn` tensor checkpoint representation.
    pub fn export_burn_state(&self) -> BurnMoeTensorState {
        BurnMoeTensorState {
            framework: "tracel-ai/burn".to_string(),
            tensor_shape: [6, 7],
            top_k: self.top_k,
            weights: self.weights,
            bias: self.bias,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_unified_diff_to_diff_hunk_expert() {
        let router = BurnMoeGatingRouter::new(2);
        let diff = "\
diff --git a/src/lib.rs b/src/lib.rs
--- a/src/lib.rs
+++ b/src/lib.rs
@@ -1,4 +1,4 @@
-fn old() {}
+fn new() {}
";
        let dec = router.route(diff);
        assert_eq!(dec.primary_expert, MoeExpertId::DiffHunkExpert);
    }

    #[test]
    fn routes_json_array_to_json_schema_expert() {
        let router = BurnMoeGatingRouter::new(2);
        let payload = r#"[{"id": 1, "status": "ok"}, {"id": 2, "status": "ok"}]"#;
        let dec = router.route(payload);
        assert_eq!(dec.primary_expert, MoeExpertId::JsonSchemaExpert);
    }

    #[test]
    fn bandit_penalty_reduces_expert_bias() {
        let mut router = BurnMoeGatingRouter::new(2);
        let before = router.export_burn_state().bias[2];
        router.update_bandit_feedback(MoeExpertId::JsonSchemaExpert, -1.0);
        let after = router.export_burn_state().bias[2];
        assert!(after < before);
    }

    #[test]
    fn routing_decision_emits_revision_sensitive_projection_certificate() {
        let router = BurnMoeGatingRouter::new(2);
        let payload = r#"[{"id": 1, "status": "ok"}, {"id": 2, "status": "ok"}]"#;
        let dec = router.route(payload);
        assert_eq!(dec.certificate.projection_operator, "pi:json_schema_expert");
        assert!(dec
            .certificate
            .retained_subspace
            .contains(&"json_envelope_schema".to_string()));
        assert!(dec
            .certificate
            .admissible_revisions
            .contains(&"R_schema_inference".to_string()));
        assert!(dec
            .certificate
            .regrounding_required_for
            .contains(&"R_exhaustive_aggregation_sum_count".to_string()));
        assert!(dec.certificate.monotone_kernel_guard);
    }
}
