"""Knowledge-worker compatible audit/export helpers for Headroom memory.

The output mirrors the small JSON graph contract used by ``mykg``:

* ``nodes`` is a mapping of node id to source/claim/entity records.
* ``edges`` is a list of typed relationships.
* durable claims point back to ``source`` nodes through ``MENTIONED_IN`` edges
  with ``source_id``, ``excerpt``, and ``confidence`` fields.

This module is intentionally dependency-free. Users who have knowledge-worker
installed can feed ``--format knowledge-worker`` output to ``mykg audit`` or
``mykg context``; Headroom can also produce a lightweight built-in audit for
the same graph shape.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from headroom.memory.adapters.graph_models import Entity, Relationship
from headroom.memory.adapters.sqlite import SQLiteMemoryStore
from headroom.memory.adapters.sqlite_graph import SQLiteGraphStore
from headroom.memory.models import Memory
from headroom.memory.ports import MemoryFilter

_KW_NODE_TYPES = {
    "person",
    "topic",
    "idea",
    "project",
    "goal",
    "question",
    "decision",
    "reference",
    "source",
}
_PROVENANCE_EDGE_TYPES = {"MENTIONED_IN", "MADE_AT"}
_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


def derive_graph_db_path(db_path: str | Path) -> Path:
    """Return the default graph DB path derived from a memory DB path."""

    path = Path(db_path)
    return path.parent / f"{path.stem}_graph{path.suffix}"


def _infer_project_from_db_path(db_path: str | Path | None) -> str | None:
    """Return the project key when db_path is inside a per-project directory."""

    if db_path is None:
        return None
    path = Path(db_path)
    if path.parent.parent.name == "projects" and path.parent.name:
        return path.parent.name
    return None


def _memory_project_keys(metadata: dict[str, Any]) -> list[str]:
    candidates = [
        metadata.get("workspace_key"),
        metadata.get("workspace_display"),
        metadata.get("project_key"),
        metadata.get("project"),
    ]
    return [
        str(candidate).strip().lower()
        for candidate in candidates
        if isinstance(candidate, str) and str(candidate).strip()
    ]


def filter_memories(
    memories: list[Memory],
    *,
    scope: str = "global",
    session_id: str | None = None,
    project: str | None = None,
    user_id: str | None = None,
    db_path: str | Path | None = None,
) -> list[Memory]:
    """Apply the proposed project/session/global export scope."""

    scoped = memories
    if user_id:
        scoped = [memory for memory in scoped if memory.user_id == user_id]

    normalized_scope = scope.lower()
    if normalized_scope == "global":
        return scoped

    if normalized_scope == "session":
        if not session_id or not session_id.strip():
            raise ValueError("session scope requires --session / session_id")
        return [memory for memory in scoped if memory.session_id == session_id]

    if normalized_scope == "project":
        if project and project.strip():
            return [
                memory for memory in scoped if _metadata_matches_project(memory.metadata, project)
            ]

        path_project = _infer_project_from_db_path(db_path)
        if not path_project:
            raise ValueError(
                "project scope requires --project when the database path is not project-specific"
            )

        # In a project-specific DB path (.../projects/<project_key>/memory.db),
        # keep rows matching that project key or untagged rows whose boundary is
        # established by the DB path itself, while excluding rows tagged for a
        # different workspace.
        return [
            memory
            for memory in scoped
            if _metadata_matches_project(memory.metadata, path_project)
            or not _memory_project_keys(memory.metadata)
        ]

    raise ValueError(f"unknown memory export scope: {scope}")


async def load_memories(
    store: SQLiteMemoryStore,
    *,
    include_superseded: bool = False,
) -> list[Memory]:
    """Load memories for audit/export."""

    memories = await store.query(MemoryFilter(include_superseded=include_superseded))
    if not include_superseded:
        memories = [
            memory
            for memory in memories
            if memory.superseded_by is None and memory.valid_until is None
        ]
    return memories


async def load_graph_parts(
    graph_store: SQLiteGraphStore | None,
    user_ids: set[str],
) -> tuple[list[Entity], list[Relationship]]:
    """Load graph entities and relationships for the selected users."""

    if graph_store is None:
        return [], []

    entities: list[Entity] = []
    relationships: list[Relationship] = []
    for user_id in sorted(user_ids):
        entities.extend(await graph_store.get_entities_for_user(user_id))
        relationships.extend(await graph_store.get_relationships_for_user(user_id))
    return entities, relationships


async def build_knowledge_worker_graph(
    store: SQLiteMemoryStore,
    *,
    graph_store: SQLiteGraphStore | None = None,
    scope: str = "global",
    session_id: str | None = None,
    project: str | None = None,
    user_id: str | None = None,
    include_superseded: bool = False,
) -> dict[str, Any]:
    """Build a knowledge-worker compatible graph from Headroom memory."""

    loaded = await load_memories(store, include_superseded=include_superseded)
    memories = filter_memories(
        loaded,
        scope=scope,
        session_id=session_id,
        project=project,
        user_id=user_id,
        db_path=getattr(store, "db_path", None),
    )
    memories_by_id = {memory.id: memory for memory in memories}
    selected_user_ids = {memory.user_id for memory in memories}
    raw_entities, raw_relationships = await load_graph_parts(graph_store, selected_user_ids)

    relationships = [
        relationship
        for relationship in raw_relationships
        if str(relationship.metadata.get("source_memory_id", "")) in memories_by_id
    ]
    referenced_entity_ids: set[str] = set()
    fallback_entity_source_memory: dict[str, Memory] = {}
    for relationship in relationships:
        rel_source_memory = memories_by_id[str(relationship.metadata.get("source_memory_id", ""))]
        referenced_entity_ids.add(relationship.source_id)
        referenced_entity_ids.add(relationship.target_id)
        fallback_entity_source_memory.setdefault(relationship.source_id, rel_source_memory)
        fallback_entity_source_memory.setdefault(relationship.target_id, rel_source_memory)

    referenced_entity_names: set[tuple[str, str]] = {
        (memory.user_id, entity_name.lower())
        for memory in memories
        for entity_name in memory.entity_refs
    }
    entities = [
        entity
        for entity in raw_entities
        if str(entity.metadata.get("source_memory_id", "")) in memories_by_id
        or entity.id in referenced_entity_ids
        or (entity.user_id, entity.name.lower()) in referenced_entity_names
    ]

    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[Any, Any, Any, Any]] = set()
    entity_id_to_kw: dict[str, str] = {}
    entity_name_to_kw: dict[tuple[str, str], str] = {}

    def add_node(record: dict[str, Any]) -> None:
        node_id = str(record["id"])
        existing = nodes.get(node_id)
        if existing is None:
            nodes[node_id] = record
            return
        # Prefer the more specific body/confidence if a graph entity and a
        # memory entity reference collide into the same concept node.
        if record.get("body") and not existing.get("body"):
            existing["body"] = record["body"]
        existing["confidence"] = _stronger_confidence(
            existing.get("confidence"), record.get("confidence")
        )

    def add_edge(record: dict[str, Any]) -> None:
        if record.get("src") in nodes and record.get("dst") in nodes:
            key = (
                record.get("src"),
                record.get("dst"),
                record.get("type"),
                record.get("source_id"),
            )
            if key in seen_edges:
                return
            seen_edges.add(key)
            edges.append(record)

    for entity in entities:
        kw_id = _entity_node_id(entity)
        entity_id_to_kw[entity.id] = kw_id
        entity_name_to_kw[(entity.user_id, entity.name.lower())] = kw_id
        source_memory = memories_by_id.get(
            str(entity.metadata.get("source_memory_id", ""))
        ) or fallback_entity_source_memory.get(entity.id)
        confidence = _metadata_confidence(entity.metadata) or (
            _memory_confidence(source_memory) if source_memory else "medium"
        )
        add_node(
            {
                "id": kw_id,
                "type": _kw_node_type(entity.entity_type),
                "label": entity.name,
                "body": entity.description or "",
                "confidence": confidence,
                "created_at": _iso(entity.created_at),
                "headroom_entity_id": entity.id,
                "headroom_entity_type": entity.entity_type,
            }
        )

    for memory in memories:
        source_id = _source_node_id(memory)
        confidence = _memory_confidence(memory)
        excerpt = _memory_excerpt(memory)
        source_text = _memory_source_text(memory)
        claim_id = _memory_claim_node_id(memory)

        add_node(
            {
                "id": source_id,
                "type": "source",
                "label": f"Headroom memory {memory.id[:8]}",
                "body": source_text,
                "confidence": "high",
                "created_at": _iso(memory.created_at),
                "headroom": {
                    "memory_id": memory.id,
                    "user_id": memory.user_id,
                    "session_id": memory.session_id,
                    "agent_id": memory.agent_id,
                    "turn_id": memory.turn_id,
                    "metadata": memory.metadata,
                },
            }
        )
        add_node(
            {
                "id": claim_id,
                "type": _infer_memory_node_type(memory),
                "label": _memory_label(memory),
                "body": memory.content,
                "confidence": confidence,
                "created_at": _iso(memory.created_at),
                "headroom_memory_id": memory.id,
            }
        )
        add_edge(
            {
                "src": claim_id,
                "dst": source_id,
                "type": "MENTIONED_IN",
                "source_id": source_id,
                "excerpt": excerpt,
                "confidence": confidence,
                "created_at": _iso(memory.created_at),
                "last_seen": _iso(memory.created_at),
            }
        )

        for entity_name in memory.entity_refs:
            entity_key = (memory.user_id, entity_name.lower())
            entity_kw_id = entity_name_to_kw.get(entity_key)
            if entity_kw_id is None:
                entity_kw_id = _entity_ref_node_id(entity_name)
                entity_name_to_kw[entity_key] = entity_kw_id
                add_node(
                    {
                        "id": entity_kw_id,
                        "type": "topic",
                        "label": entity_name,
                        "body": "",
                        "confidence": confidence,
                        "created_at": _iso(memory.created_at),
                    }
                )
            add_edge(
                {
                    "src": entity_kw_id,
                    "dst": source_id,
                    "type": "MENTIONED_IN",
                    "source_id": source_id,
                    "excerpt": excerpt,
                    "confidence": confidence,
                    "created_at": _iso(memory.created_at),
                    "last_seen": _iso(memory.created_at),
                }
            )
            add_edge(
                {
                    "src": claim_id,
                    "dst": entity_kw_id,
                    "type": "ABOUT",
                    "source_id": source_id,
                    "excerpt": excerpt,
                    "confidence": confidence,
                    "created_at": _iso(memory.created_at),
                    "last_seen": _iso(memory.created_at),
                }
            )

    for entity in entities:
        kw_id = entity_id_to_kw[entity.id]
        source_memory = memories_by_id.get(
            str(entity.metadata.get("source_memory_id", ""))
        ) or fallback_entity_source_memory.get(entity.id)
        if source_memory:
            confidence = _metadata_confidence(entity.metadata) or _memory_confidence(source_memory)
            source_id = _source_node_id(source_memory)
            add_edge(
                {
                    "src": kw_id,
                    "dst": source_id,
                    "type": "MENTIONED_IN",
                    "source_id": source_id,
                    "excerpt": _memory_excerpt(source_memory),
                    "confidence": confidence,
                    "created_at": _iso(entity.created_at),
                    "last_seen": _iso(entity.updated_at),
                }
            )

    for relationship in relationships:
        src = entity_id_to_kw.get(relationship.source_id)
        dst = entity_id_to_kw.get(relationship.target_id)
        if not src or not dst:
            continue
        source_memory = memories_by_id.get(str(relationship.metadata.get("source_memory_id", "")))
        if source_memory is None:
            continue
        source_id = _source_node_id(source_memory)
        confidence = _metadata_confidence(relationship.metadata) or _memory_confidence(
            source_memory
        )
        add_edge(
            {
                "src": src,
                "dst": dst,
                "type": _kw_edge_type(relationship.relation_type),
                "source_id": source_id,
                "excerpt": _memory_excerpt(source_memory),
                "confidence": confidence,
                "created_at": _iso(relationship.created_at),
                "last_seen": _iso(relationship.created_at),
                "headroom_relationship_id": relationship.id,
                "headroom_relation_type": relationship.relation_type,
            }
        )

    return {
        "nodes": dict(sorted(nodes.items())),
        "edges": sorted(
            edges,
            key=lambda edge: (
                str(edge.get("src", "")),
                str(edge.get("dst", "")),
                str(edge.get("type", "")),
                str(edge.get("source_id", "")),
            ),
        ),
        "_meta": {
            "schema_version": "headroom-knowledge-worker/v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": "headroom memory",
            "scope": scope,
            "session_id": session_id,
            "project": project,
            "user_id": user_id,
            "memory_count": len(memories),
        },
    }


def build_context_snapshot(graph: dict[str, Any], *, max_ideas: int = 20) -> str:
    """Render a compact Markdown handoff from a knowledge-worker graph."""

    nodes = graph.get("nodes", {})
    edges = graph.get("edges", [])
    incoming: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        incoming[str(edge.get("dst", ""))].append(edge)

    by_type_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes.values():
        by_type_index[node.get("type")].append(node)

    def by_type(node_type: str) -> list[dict[str, Any]]:
        return by_type_index.get(node_type, [])

    def marker(confidence: str | None) -> str:
        if confidence == "low":
            return " WARN low"
        if confidence == "medium":
            return " ~ medium"
        return ""

    lines = [
        "# Headroom Memory Context",
        (
            f"*Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} | "
            f"{len(nodes)} nodes, {len(edges)} edges*"
        ),
        "",
    ]

    for heading, node_type, limit in [
        ("Goals", "goal", 20),
        ("Key Decisions", "decision", 20),
        ("Open Questions", "question", 20),
    ]:
        typed = by_type(node_type)[:limit]
        if not typed:
            continue
        lines.append(f"## {heading}")
        for node in typed:
            lines.append(f"- **{node['label']}**{marker(node.get('confidence'))}")
            if node.get("body"):
                lines.append(f"  {str(node['body'])[:160]}")
        lines.append("")

    ideas = sorted(
        by_type("idea"),
        key=lambda node: (-len(incoming.get(node["id"], [])), str(node.get("label", "")).lower()),
    )
    if ideas:
        lines.append("## Ideas")
        for node in ideas[:max_ideas]:
            lines.append(
                f"- **{node['label']}**{marker(node.get('confidence'))} "
                f"*(connections: {len(incoming.get(node['id'], []))})*"
            )
            if node.get("body"):
                lines.append(f"  {str(node['body'])[:160]}")
        lines.append("")

    topics = sorted(
        by_type("topic"),
        key=lambda node: (-len(incoming.get(node["id"], [])), str(node.get("label", "")).lower()),
    )
    if topics:
        lines.append("## Core Topics")
        for node in topics[:20]:
            lines.append(f"- {node['label']} *(x{len(incoming.get(node['id'], []))})*")
        lines.append("")

    sources = sorted(
        by_type("source"),
        key=lambda node: str(node.get("created_at", "")),
        reverse=True,
    )
    if sources:
        lines.append("## Recent Sources")
        for node in sources[:10]:
            lines.append(f"- {node['label']}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_memory_audit(graph: dict[str, Any], *, limit: int = 25) -> dict[str, Any]:
    """Build deterministic audit analytics for a Headroom memory graph."""

    nodes = graph.get("nodes", {})
    edges = graph.get("edges", [])
    semantic_ids = sorted(
        node_id for node_id, node in nodes.items() if node.get("type") != "source"
    )
    semantic_set = set(semantic_ids)
    semantic_edges = [
        edge
        for edge in edges
        if edge.get("src") in semantic_set
        and edge.get("dst") in semantic_set
        and edge.get("type") not in _PROVENANCE_EDGE_TYPES
    ]
    adjacency = _adjacency(semantic_ids, semantic_edges)
    degree = {node_id: len(neighbors) for node_id, neighbors in adjacency.items()}
    coverage = _provenance_coverage(graph)
    weak_claims = _weak_claims(graph, coverage, limit)

    ranked = sorted(
        semantic_ids,
        key=lambda node_id: (
            -degree.get(node_id, 0),
            str(nodes[node_id].get("type", "")),
            str(nodes[node_id].get("label", "")).lower(),
        ),
    )

    return {
        "schema_version": "headroom-memory-audit/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stats": {
            "nodes": len(nodes),
            "edges": len(edges),
            "semantic_nodes": len(semantic_ids),
            "semantic_edges": len(semantic_edges),
            "source_nodes": sum(1 for node in nodes.values() if node.get("type") == "source"),
            "semantic_components": len(_components(semantic_ids, adjacency)),
        },
        "counts": {
            "node_types": dict(Counter(str(node.get("type", "")) for node in nodes.values())),
            "edge_types": dict(Counter(str(edge.get("type", "")) for edge in edges)),
            "confidence": {
                "nodes": dict(Counter(str(node.get("confidence", "")) for node in nodes.values())),
                "edges": dict(Counter(str(edge.get("confidence", "")) for edge in edges)),
            },
        },
        "ranked": {
            "important_concepts": [
                _node_record(graph, node_id, degree=degree.get(node_id, 0))
                for node_id in ranked[:limit]
            ],
            "weak_claims": weak_claims,
            "weak_claim_queue": _weak_claim_queue(weak_claims, limit),
        },
        "provenance_coverage": coverage,
        "meta": graph.get("_meta", {}),
    }


def _metadata_matches_project(metadata: dict[str, Any], project: str) -> bool:
    project_lower = project.lower()
    candidates = [
        metadata.get("workspace_key"),
        metadata.get("workspace_display"),
        metadata.get("project_key"),
        metadata.get("project"),
    ]
    return any(str(candidate).lower() == project_lower for candidate in candidates if candidate)


def _source_node_id(memory: Memory | None) -> str:
    if memory is None:
        return ""
    return f"source:headroom-memory-{_stable_suffix(memory.id)}"


def _memory_claim_node_id(memory: Memory) -> str:
    node_type = _infer_memory_node_type(memory)
    return f"{node_type}:headroom-memory-{_stable_suffix(memory.id)}"


def _entity_ref_node_id(entity_name: str) -> str:
    return f"topic:{_slug(entity_name)}-{_short_hash(entity_name)}"


def _entity_node_id(entity: Entity) -> str:
    node_type = _kw_node_type(entity.entity_type)
    return f"{node_type}:{_slug(entity.name)}-{_stable_suffix(entity.id)}"


def _kw_node_type(entity_type: str | None) -> str:
    normalized = (entity_type or "topic").lower().replace(" ", "_").replace("-", "_")
    mapping = {
        "organization": "project",
        "org": "project",
        "company": "project",
        "technology": "topic",
        "tech": "topic",
        "system": "topic",
        "service": "topic",
        "component": "topic",
        "module": "topic",
        "concept": "idea",
        "preference": "idea",
    }
    mapped = mapping.get(normalized, normalized)
    return mapped if mapped in _KW_NODE_TYPES else "topic"


def _kw_edge_type(relation_type: str | None) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", relation_type or "RELATES_TO")
    normalized = normalized.strip("_").upper()
    return normalized or "RELATES_TO"


def _infer_memory_node_type(memory: Memory) -> str:
    kind = str(memory.metadata.get("kind") or memory.metadata.get("category") or "").lower()
    if kind in _KW_NODE_TYPES and kind != "source":
        return kind
    text = memory.content.strip().lower()
    if text.endswith("?") or text.startswith(("question:", "open question")):
        return "question"
    if text.startswith(("decided", "decision:", "we chose", "chose ")):
        return "decision"
    if text.startswith(("goal:", "objective:", "target:")):
        return "goal"
    if memory.importance >= 0.85:
        return "idea"
    return "idea"


def _memory_label(memory: Memory) -> str:
    stripped = " ".join(memory.content.split())
    return stripped[:76] + "..." if len(stripped) > 79 else stripped or memory.id[:8]


def _memory_source_text(memory: Memory) -> str:
    source_text = memory.metadata.get("source_text")
    for alt_key in ("source_chunk", "source"):
        alt_val = memory.metadata.get(alt_key)
        if (
            isinstance(alt_val, str)
            and alt_val.strip()
            and (
                not isinstance(source_text, str)
                or not source_text.strip()
                or source_text == memory.content
            )
        ):
            return alt_val
    for key in ("source_text", "source_chunk", "source"):
        value = memory.metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return memory.content


def _memory_excerpt(memory: Memory | None) -> str:
    if memory is None:
        return ""
    if memory.metadata.get("source_is_synthetic"):
        return ""
    source_excerpt = memory.metadata.get("source_excerpt")
    for alt_key in ("excerpt", "source_chunk"):
        alt_val = memory.metadata.get(alt_key)
        if (
            isinstance(alt_val, str)
            and alt_val.strip()
            and (
                not isinstance(source_excerpt, str)
                or not source_excerpt.strip()
                or source_excerpt == memory.content[:1000]
            )
        ):
            return alt_val[:1000]
    for key in ("source_excerpt", "excerpt", "source_chunk"):
        value = memory.metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value[:1000]
    if "source_excerpt" in memory.metadata and memory.metadata.get("source_excerpt") == "":
        return ""
    return memory.content[:1000]


def _memory_confidence(memory: Memory | None) -> str:
    if memory is None:
        return "medium"
    metadata_confidence = _metadata_confidence(memory.metadata)
    if metadata_confidence:
        return metadata_confidence
    if memory.importance >= 0.8:
        return "high"
    if memory.importance >= 0.4:
        return "medium"
    return "low"


def _metadata_confidence(metadata: dict[str, Any]) -> str | None:
    raw = metadata.get("confidence") or metadata.get("headroom_confidence")
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        if raw >= 0.8:
            return "high"
        if raw >= 0.4:
            return "medium"
        return "low"
    normalized = str(raw).lower().strip()
    if normalized in _CONFIDENCE_RANK:
        return normalized
    return None


def _stronger_confidence(left: Any, right: Any) -> str:
    left_norm = str(left or "medium").lower()
    right_norm = str(right or "medium").lower()
    return (
        left_norm
        if _CONFIDENCE_RANK.get(left_norm, 1) >= _CONFIDENCE_RANK.get(right_norm, 1)
        else right_norm
    )


def _provenance_coverage(graph: dict[str, Any]) -> dict[str, Any]:
    nodes = graph.get("nodes", {})
    edges = graph.get("edges", [])
    non_source_ids = {node_id for node_id, node in nodes.items() if node.get("type") != "source"}
    mentioned: set[str] = set()
    mentioned_with_excerpt: set[str] = set()
    provenance_edges: list[dict[str, Any]] = []
    for edge in edges:
        if edge.get("type") not in _PROVENANCE_EDGE_TYPES:
            continue
        provenance_edges.append(edge)
        for endpoint in (edge.get("src"), edge.get("dst")):
            if endpoint in non_source_ids:
                mentioned.add(str(endpoint))
                if edge.get("excerpt"):
                    mentioned_with_excerpt.add(str(endpoint))

    missing_nodes = sorted(non_source_ids - mentioned)
    edges_with_source_id = [edge for edge in edges if edge.get("source_id")]
    edges_missing_source_id = [
        {**edge, "index": index} for index, edge in enumerate(edges) if not edge.get("source_id")
    ]
    provenance_with_excerpt = [edge for edge in provenance_edges if edge.get("excerpt")]

    def ratio(numerator: int, denominator: int) -> float:
        return 1.0 if denominator == 0 else numerator / denominator

    return {
        "node_coverage": ratio(len(mentioned), len(non_source_ids)),
        "excerpt_coverage": ratio(len(provenance_with_excerpt), len(provenance_edges)),
        "edge_source_coverage": ratio(len(edges_with_source_id), len(edges)),
        "non_source_nodes": len(non_source_ids),
        "nodes_with_provenance": len(mentioned),
        "nodes_with_provenance_excerpt": len(mentioned_with_excerpt),
        "missing_nodes": [_node_record(graph, node_id) for node_id in missing_nodes],
        "edges_total": len(edges),
        "edges_with_source_id": len(edges_with_source_id),
        "edges_missing_source_id": edges_missing_source_id,
        "provenance_edges": len(provenance_edges),
        "provenance_edges_with_excerpt": len(provenance_with_excerpt),
    }


def _weak_claims(
    graph: dict[str, Any],
    coverage: dict[str, Any],
    limit: int,
) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for node_id, node in graph.get("nodes", {}).items():
        if node.get("type") == "source" or not _confidence_is_weak(node.get("confidence")):
            continue
        claims.append({"kind": "node_confidence", **_node_record(graph, node_id)})
    for index, edge in enumerate(graph.get("edges", [])):
        if _confidence_is_weak(edge.get("confidence")):
            claims.append({"kind": "edge_confidence", **edge, "index": index})
    for node in coverage.get("missing_nodes", []):
        claims.append({"kind": "missing_node_provenance", **node})
    for edge in coverage.get("edges_missing_source_id", []):
        claims.append({"kind": "missing_edge_source_id", **edge})

    return sorted(
        claims,
        key=lambda claim: (
            _CONFIDENCE_RANK.get(str(claim.get("confidence", "")), -1),
            str(claim.get("kind", "")),
            str(claim.get("id") or claim.get("src") or ""),
        ),
    )[:limit]


def _weak_claim_queue(claims: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    queue = []
    for claim in claims[:limit]:
        prompt = "Choose: verify, downgrade, convert to question, ignore for now."
        if claim.get("kind") == "missing_node_provenance":
            prompt = "Find source evidence or keep this out of durable memory."
        elif claim.get("kind") == "missing_edge_source_id":
            prompt = "Attach a source id or remove this edge from the durable graph."
        elif claim.get("kind") == "edge_confidence":
            prompt = "Inspect this relationship: verify it, downgrade it, or turn it into an open question."
        queue.append(
            {
                **claim,
                "prompt": prompt,
                "review_options": [
                    "verify",
                    "downgrade",
                    "convert_to_question",
                    "ignore_for_now",
                ],
            }
        )
    return queue


def _confidence_is_weak(confidence: Any) -> bool:
    return str(confidence or "high").lower() != "high"


def _node_record(graph: dict[str, Any], node_id: str, **metrics: Any) -> dict[str, Any]:
    node = graph.get("nodes", {}).get(node_id, {})
    record = {
        "id": node_id,
        "type": node.get("type"),
        "label": node.get("label"),
        "confidence": node.get("confidence"),
    }
    record.update(metrics)
    return record


def _adjacency(
    node_ids: list[str],
    edges: list[dict[str, Any]],
) -> dict[str, set[str]]:
    adjacency = {node_id: set() for node_id in node_ids}
    for edge in edges:
        src = str(edge.get("src", ""))
        dst = str(edge.get("dst", ""))
        if src in adjacency and dst in adjacency:
            adjacency[src].add(dst)
            adjacency[dst].add(src)
    return adjacency


def _components(
    node_ids: list[str],
    adjacency: dict[str, set[str]],
) -> list[list[str]]:
    remaining = set(node_ids)
    components: list[list[str]] = []
    for start in sorted(node_ids):
        if start not in remaining:
            continue
        remaining.discard(start)
        queue = deque([start])
        component: list[str] = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor in remaining:
                    remaining.discard(neighbor)
                    queue.append(neighbor)
        components.append(sorted(component))
    return components


@lru_cache(maxsize=8192)
def _slug(value: str) -> str:
    slugged = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slugged or "item"


@lru_cache(maxsize=8192)
def _stable_suffix(value: str) -> str:
    cleaned = _slug(value)
    if not cleaned:
        return _short_hash(value)
    if len(cleaned) <= 32 and cleaned == value.lower().strip("-"):
        return cleaned
    return f"{cleaned[:24]}-{_short_hash(value)}"


@lru_cache(maxsize=8192)
def _short_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]


def _iso(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)
