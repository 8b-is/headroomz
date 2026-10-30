"""Tests for the Sparse Mixture-of-Experts (MOEOptimizer), Burn tensor gating, and Iroh P2P mesh."""

from __future__ import annotations

import json

from headroom.proxy.helpers import (
    apply_moe_to_messages,
    get_moe_optimizer_stats,
    moe_optimizer_enabled,
    reset_moe_optimizer_stats,
)
from headroom.telemetry.iroh_mesh import (
    ALPN_MOE_BURN_GOSSIP,
    IrohMeshNode,
)
from headroom.transforms.moe_optimizer import (
    ExpertId,
    MOEGatingNetwork,
    MOEOptimizer,
)


def test_gating_routes_json_array_to_json_schema_expert() -> None:
    gating = MOEGatingNetwork(top_k=2)
    items = [{"id": i, "status": "ok", "service": f"svc-{i}"} for i in range(40)]
    payload = json.dumps(items, indent=2)
    decision = gating.route(payload)
    assert decision.primary_expert == ExpertId.JSON_SCHEMA
    assert decision.primary_confidence > 0.5


def test_gating_routes_unified_diff_to_diff_hunk_expert() -> None:
    gating = MOEGatingNetwork(top_k=2)
    diff_lines = [
        "diff --git a/headroom/proxy/server.py b/headroom/proxy/server.py",
        "--- a/headroom/proxy/server.py",
        "+++ b/headroom/proxy/server.py",
        "@@ -100,10 +100,12 @@",
        "-    old_call()",
        "+    new_call()",
    ] + [f"+    added_line_{i} = {i}" for i in range(30)]
    decision = gating.route("\n".join(diff_lines))
    assert decision.primary_expert == ExpertId.DIFF_HUNK


def test_gating_routes_markdown_table_to_tabular_expert() -> None:
    gating = MOEGatingNetwork(top_k=2)
    rows = [
        "| id | host | status | latency_ms |",
        "| --- | --- | --- | --- |",
    ] + [f"| {i} | node-{i} | healthy | 12.{i} |" for i in range(45)]
    decision = gating.route("\n".join(rows))
    assert decision.primary_expert == ExpertId.TABULAR


def test_moe_optimizer_compresses_json_array_while_preserving_errors() -> None:
    moe = MOEOptimizer(min_lines=10)
    records = [
        {"id": i, "status": "healthy" if i != 22 else "ERROR: connection timeout", "latency": i}
        for i in range(60)
    ]
    raw = json.dumps(records, indent=2)
    res = moe.optimize(raw, record_proxy_stats=True)
    assert res.primary_expert == ExpertId.JSON_SCHEMA.value
    assert res.tokens_saved > 0
    assert "ERROR: connection timeout" in res.text


def test_moe_optimizer_compresses_repetitive_logs() -> None:
    moe = MOEOptimizer(min_lines=10)
    lines = [f"2026-10-30 12:00:{i % 60:02d} INFO worker heartbeat item={i} ok" for i in range(80)]
    lines.insert(40, "2026-10-30 12:00:40 ERROR database deadlock detected on shard 3")
    raw = "\n".join(lines)
    res = moe.optimize(raw, record_proxy_stats=True)
    assert res.tokens_saved > 0
    assert "ERROR database deadlock detected" in res.text


def test_burn_tensor_state_and_bandit_feedback() -> None:
    moe = MOEOptimizer()
    state_before = moe.gating.export_burn_tensor_state()
    assert state_before["framework"] == "tracel-ai/burn"
    assert state_before["tensor_shape"] == [6, 7]
    bias_before = state_before["expert_bias"][ExpertId.JSON_SCHEMA.value]

    moe.record_retrieval_penalty(ExpertId.JSON_SCHEMA.value)
    state_after = moe.gating.export_burn_tensor_state()
    assert state_after["expert_bias"][ExpertId.JSON_SCHEMA.value] < bias_before
    assert moe.get_stats()["retrieval_penalties"] == 1


def test_iroh_mesh_blob_ticket_and_federated_gossip() -> None:
    node = IrohMeshNode()
    status = node.get_status()
    assert status["provider"] == "n0-computer/iroh"
    assert status["node_id"].startswith("iroh1")
    assert status["burn_checkpoint_ticket"].startswith("bloba")

    blob = node.publish_blob(
        "compressed payload",
        kind="moe_test",
        expert=ExpertId.DIFF_HUNK.value,
        tokens_saved=128,
    )
    assert blob["ticket"].startswith("bloba")
    assert len(blob["content_hash"]) == 64

    gossip = node.sync_moe_gossip({"diff_hunk_expert": 0.4})
    assert gossip["status"] == "synced"
    assert gossip["protocol"] == ALPN_MOE_BURN_GOSSIP
    assert gossip["checkpoint_ticket"].startswith("bloba")


def test_apply_moe_to_messages_updates_proxy_stats() -> None:
    reset_moe_optimizer_stats()
    assert moe_optimizer_enabled({"x-headroom-moe-optimizer": "true"}) is True
    assert moe_optimizer_enabled({"x-headroom-moe-optimizer": "false"}) is False

    big_table = "\n".join(
        ["| idx | metric | state | note |", "| --- | --- | --- | --- |"]
        + [f"| {i} | cpu_load | nominal | worker_{i}_ok |" for i in range(80)]
    )
    messages = [
        {"role": "user", "content": "check cluster table"},
        {"role": "tool", "content": big_table},
    ]
    out, dropped, expert = apply_moe_to_messages(messages, window_radius=15)
    assert len(out[1]["content"]) < len(big_table)
    assert dropped > 0
    assert expert == ExpertId.TABULAR.value
    stats = get_moe_optimizer_stats()
    assert stats["proxy_calls"] >= 1
    assert stats["proxy_tokens_saved"] > 0
    assert "iroh_mesh" in stats
    assert "burn_gating" in stats
    assert "projection_certificates" in stats


def test_emits_revision_sensitive_projection_certificate() -> None:
    moe = MOEOptimizer(window_radius=8, top_k=2, min_lines=10)
    items = [{"id": i, "status": "ok", "latency_ms": 11} for i in range(40)]
    payload = json.dumps(items, indent=2)
    res = moe.optimize(payload)
    cert = res.certificate
    assert cert["certificate_version"] == "1.1-sufficient-projection"
    assert cert["projection_operator"] == "pi:json_schema_expert"
    assert "json_envelope_schema" in cert["retained_subspace"]
    assert "R_schema_inference" in cert["admissible_revisions"]
    assert "R_exhaustive_aggregation_sum_count" in cert["regrounding_required_for"]
    assert cert["monotone_kernel_guard"] is True
    assert cert["idempotent_skip"] is False
    assert cert["bennett_reversible_handle"] == "ccr://headroom_retrieve"


def test_monotone_kernel_growth_guard_prevents_iterative_reprojection() -> None:
    moe = MOEOptimizer(window_radius=8, top_k=2, min_lines=10)
    logs = [f"2026-10-30T12:00:{i % 60:02d} INFO worker={i} heartbeat ok" for i in range(60)]
    turn_1 = moe.optimize("\n".join(logs))
    assert turn_1.dropped_lines > 0
    assert turn_1.certificate["idempotent_skip"] is False

    # Passing the already-projected output into a second turn must be idempotent:
    turn_2 = moe.optimize(turn_1.text)
    assert turn_2.text == turn_1.text
    assert turn_2.dropped_lines == 0
    assert turn_2.certificate["idempotent_skip"] is True
    assert moe.get_stats()["monotone_kernel_skips"] >= 1
