"""Tests for memory CLI commands.

These tests use real SQLite databases (temp files) - no mocks.
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from click.testing import CliRunner

from headroom.cli.main import main
from headroom.memory.adapters.sqlite import SQLiteMemoryStore
from headroom.memory.backends.local import _enrich_memory_metadata
from headroom.memory.models import Memory


@pytest.fixture
def runner() -> CliRunner:
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_db(tmp_path: Path) -> str:
    """Create a temporary database path."""
    return str(tmp_path / "test_memory.db")


@pytest.fixture
def populated_db(temp_db: str) -> str:
    """Create a database with sample memories."""
    store = SQLiteMemoryStore(temp_db)

    # Create memories at different scopes and ages
    memories = [
        # USER scope (no session/agent/turn)
        Memory(
            id="user-mem-001",
            content="User prefers TypeScript over JavaScript",
            user_id="test-user",
            session_id=None,
            agent_id=None,
            turn_id=None,
            importance=0.9,
            created_at=datetime.now() - timedelta(days=5),
            valid_from=datetime.now() - timedelta(days=5),
        ),
        # SESSION scope
        Memory(
            id="session-mem-001",
            content="Working on authentication feature",
            user_id="test-user",
            session_id="session-123",
            agent_id=None,
            turn_id=None,
            importance=0.7,
            created_at=datetime.now() - timedelta(hours=2),
            valid_from=datetime.now() - timedelta(hours=2),
        ),
        Memory(
            id="session-mem-002",
            content="Database uses PostgreSQL",
            user_id="test-user",
            session_id="session-123",
            agent_id=None,
            turn_id=None,
            importance=0.6,
            created_at=datetime.now() - timedelta(days=10),
            valid_from=datetime.now() - timedelta(days=10),
        ),
        # AGENT scope
        Memory(
            id="agent-mem-001",
            content="Agent is exploring code structure",
            user_id="test-user",
            session_id="session-123",
            agent_id="agent-456",
            turn_id=None,
            importance=0.4,
            created_at=datetime.now() - timedelta(hours=1),
            valid_from=datetime.now() - timedelta(hours=1),
        ),
        # TURN scope (ephemeral)
        Memory(
            id="turn-mem-001",
            content="Tool output from grep search",
            user_id="test-user",
            session_id="session-123",
            agent_id="agent-456",
            turn_id="turn-789",
            importance=0.2,
            created_at=datetime.now() - timedelta(minutes=5),
            valid_from=datetime.now() - timedelta(minutes=5),
        ),
        # Low importance memory for pruning tests
        Memory(
            id="low-importance-001",
            content="Temporary note",
            user_id="test-user",
            session_id="session-123",
            agent_id=None,
            turn_id=None,
            importance=0.1,
            created_at=datetime.now() - timedelta(days=45),
            valid_from=datetime.now() - timedelta(days=45),
        ),
    ]

    for mem in memories:
        asyncio.run(store.save(mem))

    return temp_db


class TestMemoryList:
    """Tests for 'headroom memory list' command."""

    def test_list_all(self, runner: CliRunner, populated_db: str) -> None:
        """List all memories."""
        result = runner.invoke(main, ["memory", "list", "--db-path", populated_db])
        assert result.exit_code == 0
        # IDs are truncated to 8 chars in display, check for partial matches
        assert "user-mem" in result.output
        assert "session-" in result.output  # "session-mem" truncated to "session-"

    def test_list_with_limit(self, runner: CliRunner, populated_db: str) -> None:
        """List with limit."""
        result = runner.invoke(main, ["memory", "list", "--db-path", populated_db, "--limit", "2"])
        assert result.exit_code == 0
        # Should show limited results
        assert "2 shown" in result.output or "Memories" in result.output

    def test_list_by_scope(self, runner: CliRunner, populated_db: str) -> None:
        """Filter by scope level."""
        result = runner.invoke(
            main, ["memory", "list", "--db-path", populated_db, "--scope", "USER"]
        )
        assert result.exit_code == 0
        assert "TypeScript" in result.output  # USER scope memory content

    def test_list_empty_db(self, runner: CliRunner, temp_db: str) -> None:
        """List from empty database."""
        # Initialize empty db
        SQLiteMemoryStore(temp_db)
        result = runner.invoke(main, ["memory", "list", "--db-path", temp_db])
        assert result.exit_code == 0
        assert "No memories found" in result.output


class TestMemoryShow:
    """Tests for 'headroom memory show' command."""

    def test_show_by_id(self, runner: CliRunner, populated_db: str) -> None:
        """Show memory by full ID."""
        result = runner.invoke(main, ["memory", "show", "--db-path", populated_db, "user-mem-001"])
        assert result.exit_code == 0
        assert "TypeScript" in result.output
        assert "0.9" in result.output or "0.90" in result.output  # importance

    def test_show_by_partial_id(self, runner: CliRunner, populated_db: str) -> None:
        """Show memory by partial ID."""
        result = runner.invoke(main, ["memory", "show", "--db-path", populated_db, "user-mem"])
        assert result.exit_code == 0
        assert "TypeScript" in result.output

    def test_show_by_partial_id_beyond_default_query_limit(
        self, runner: CliRunner, temp_db: str
    ) -> None:
        """Show a unique partial ID when the match is older than 1,000 rows."""
        store = SQLiteMemoryStore(temp_db)
        filler_start = datetime(2024, 1, 1)
        memories = [
            Memory(
                id=f"filler-{index:04d}",
                content=f"Filler memory {index}",
                user_id="test-user",
                created_at=filler_start + timedelta(days=index),
                valid_from=filler_start + timedelta(days=index),
            )
            for index in range(1000)
        ]
        memories.append(
            Memory(
                id="legacy-target-memory",
                content="Legacy target outside the first page",
                user_id="test-user",
                created_at=datetime(2020, 1, 1),
                valid_from=datetime(2020, 1, 1),
            )
        )
        asyncio.run(store.save_batch(memories))

        result = runner.invoke(main, ["memory", "show", "--db-path", temp_db, "legacy-target-"])

        assert result.exit_code == 0
        assert "Legacy target outside the first page" in result.output

    def test_show_by_partial_id_reports_ambiguity_beyond_default_query_limit(
        self, runner: CliRunner, temp_db: str
    ) -> None:
        """Report ambiguity when matching IDs span beyond the first 1,000 rows."""
        store = SQLiteMemoryStore(temp_db)
        filler_start = datetime(2024, 1, 1)
        memories = [
            Memory(
                id=f"filler-{index:04d}",
                content=f"Filler memory {index}",
                user_id="test-user",
                created_at=filler_start + timedelta(days=index),
                valid_from=filler_start + timedelta(days=index),
            )
            for index in range(1000)
        ]
        memories.extend(
            [
                Memory(
                    id="shared-prefix-old",
                    content="Older matching memory",
                    user_id="test-user",
                    created_at=datetime(2020, 1, 1),
                    valid_from=datetime(2020, 1, 1),
                ),
                Memory(
                    id="shared-prefix-new",
                    content="Newer matching memory",
                    user_id="test-user",
                    created_at=datetime(2026, 1, 1),
                    valid_from=datetime(2026, 1, 1),
                ),
            ]
        )
        asyncio.run(store.save_batch(memories))

        result = runner.invoke(main, ["memory", "show", "--db-path", temp_db, "shared-prefix-"])

        assert result.exit_code != 0
        assert "Ambiguous ID 'shared-prefix-'" in result.output

    def test_show_json_output(self, runner: CliRunner, populated_db: str) -> None:
        """Show memory as JSON."""
        result = runner.invoke(
            main, ["memory", "show", "--db-path", populated_db, "user-mem-001", "--json"]
        )
        assert result.exit_code == 0
        # Should be valid JSON
        data = json.loads(result.output)
        assert data["id"] == "user-mem-001"
        assert "TypeScript" in data["content"]

    def test_show_not_found(self, runner: CliRunner, populated_db: str) -> None:
        """Show non-existent memory."""
        result = runner.invoke(
            main, ["memory", "show", "--db-path", populated_db, "nonexistent-id"]
        )
        assert result.exit_code != 0 or "not found" in result.output.lower()


class TestMemoryStats:
    """Tests for 'headroom memory stats' command."""

    def test_stats(self, runner: CliRunner, populated_db: str) -> None:
        """Show stats for populated database."""
        result = runner.invoke(main, ["memory", "stats", "--db-path", populated_db])
        assert result.exit_code == 0
        assert "Total" in result.output or "Memories" in result.output
        assert "6" in result.output  # 6 memories

    def test_stats_empty_db(self, runner: CliRunner, temp_db: str) -> None:
        """Stats for empty database."""
        SQLiteMemoryStore(temp_db)
        result = runner.invoke(main, ["memory", "stats", "--db-path", temp_db])
        assert result.exit_code == 0
        assert "0" in result.output


class TestMemoryEdit:
    """Tests for 'headroom memory edit' command."""

    def test_edit_content(self, runner: CliRunner, populated_db: str) -> None:
        """Edit memory content."""
        result = runner.invoke(
            main,
            [
                "memory",
                "edit",
                "--db-path",
                populated_db,
                "user-mem-001",
                "--content",
                "Updated content",
            ],
        )
        assert result.exit_code == 0

        # Verify change
        show_result = runner.invoke(
            main, ["memory", "show", "--db-path", populated_db, "user-mem-001"]
        )
        assert "Updated content" in show_result.output

    def test_edit_importance(self, runner: CliRunner, populated_db: str) -> None:
        """Edit memory importance."""
        result = runner.invoke(
            main,
            ["memory", "edit", "--db-path", populated_db, "user-mem-001", "--importance", "0.5"],
        )
        assert result.exit_code == 0

        # Verify change
        show_result = runner.invoke(
            main, ["memory", "show", "--db-path", populated_db, "user-mem-001"]
        )
        assert "0.5" in show_result.output

    def test_edit_not_found(self, runner: CliRunner, populated_db: str) -> None:
        """Edit non-existent memory."""
        result = runner.invoke(
            main,
            ["memory", "edit", "--db-path", populated_db, "nonexistent", "--content", "test"],
        )
        assert result.exit_code != 0 or "not found" in result.output.lower()


class TestMemoryDelete:
    """Tests for 'headroom memory delete' command."""

    def test_delete_single(self, runner: CliRunner, populated_db: str) -> None:
        """Delete single memory with force."""
        result = runner.invoke(
            main,
            ["memory", "delete", "--db-path", populated_db, "turn-mem-001", "--force"],
        )
        assert result.exit_code == 0

        # Verify deleted
        show_result = runner.invoke(
            main, ["memory", "show", "--db-path", populated_db, "turn-mem-001"]
        )
        assert "not found" in show_result.output.lower() or show_result.exit_code != 0

    def test_delete_multiple(self, runner: CliRunner, populated_db: str) -> None:
        """Delete multiple memories."""
        result = runner.invoke(
            main,
            [
                "memory",
                "delete",
                "--db-path",
                populated_db,
                "turn-mem-001",
                "agent-mem-001",
                "--force",
            ],
        )
        assert result.exit_code == 0

    def test_delete_requires_confirmation(self, runner: CliRunner, populated_db: str) -> None:
        """Delete prompts for confirmation without --force."""
        # Invoke delete and say no to confirmation
        runner.invoke(
            main,
            ["memory", "delete", "--db-path", populated_db, "turn-mem-001"],
            input="n\n",  # Say no
        )
        # Verify memory still exists since we said no
        show_result = runner.invoke(
            main, ["memory", "show", "--db-path", populated_db, "turn-mem-001"]
        )
        # Memory should still exist since we said no
        assert "Tool output" in show_result.output or show_result.exit_code == 0


class TestMemoryPrune:
    """Tests for 'headroom memory prune' command."""

    def test_prune_dry_run(self, runner: CliRunner, populated_db: str) -> None:
        """Prune with dry-run shows what would be deleted."""
        result = runner.invoke(
            main,
            ["memory", "prune", "--db-path", populated_db, "--older-than", "30d", "--dry-run"],
        )
        assert result.exit_code == 0
        assert "would" in result.output.lower() or "dry" in result.output.lower()

    def test_prune_by_age(self, runner: CliRunner, populated_db: str) -> None:
        """Prune old memories."""
        result = runner.invoke(
            main,
            ["memory", "prune", "--db-path", populated_db, "--older-than", "30d", "--force"],
        )
        assert result.exit_code == 0
        # Should have deleted the 45-day old memory

    def test_prune_by_scope(self, runner: CliRunner, populated_db: str) -> None:
        """Prune by scope level."""
        result = runner.invoke(
            main,
            ["memory", "prune", "--db-path", populated_db, "--scope", "TURN", "--force"],
        )
        assert result.exit_code == 0

        # Verify TURN memories are gone
        list_result = runner.invoke(
            main, ["memory", "list", "--db-path", populated_db, "--scope", "TURN"]
        )
        assert "No memories found" in list_result.output or "turn-mem" not in list_result.output

    def test_prune_low_importance(self, runner: CliRunner, populated_db: str) -> None:
        """Prune low importance memories."""
        result = runner.invoke(
            main,
            ["memory", "prune", "--db-path", populated_db, "--low-importance", "0.3", "--force"],
        )
        assert result.exit_code == 0


class TestMemoryPurge:
    """Tests for 'headroom memory purge' command."""

    def test_purge_requires_confirm_flag(self, runner: CliRunner, populated_db: str) -> None:
        """Purge requires --confirm flag."""
        result = runner.invoke(main, ["memory", "purge", "--db-path", populated_db])
        assert result.exit_code != 0 or "confirm" in result.output.lower()

    def test_purge_with_confirm(self, runner: CliRunner, populated_db: str) -> None:
        """Purge deletes all memories."""
        result = runner.invoke(
            main,
            ["memory", "purge", "--db-path", populated_db, "--confirm"],
            input="y\n",  # Confirm
        )
        assert result.exit_code == 0

        # Verify empty
        stats_result = runner.invoke(main, ["memory", "stats", "--db-path", populated_db])
        assert "0" in stats_result.output


class TestMemoryExportImport:
    """Tests for export/import commands."""

    def test_export_to_stdout(self, runner: CliRunner, populated_db: str) -> None:
        """Export memories to stdout."""
        result = runner.invoke(main, ["memory", "export", "--db-path", populated_db])
        assert result.exit_code == 0
        # Should be valid JSON array
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) == 6

    def test_export_to_file(self, runner: CliRunner, populated_db: str, tmp_path: Path) -> None:
        """Export memories to file."""
        output_file = tmp_path / "export.json"
        result = runner.invoke(
            main,
            ["memory", "export", "--db-path", populated_db, "--output", str(output_file)],
        )
        assert result.exit_code == 0

        # Verify file
        with open(output_file) as f:
            data = json.load(f)
        assert len(data) == 6

    def test_export_knowledge_worker_scoped_to_session(
        self, runner: CliRunner, populated_db: str
    ) -> None:
        """Knowledge-worker export emits provenance graph JSON for a session."""
        result = runner.invoke(
            main,
            [
                "memory",
                "export",
                "--db-path",
                populated_db,
                "--format",
                "knowledge-worker",
                "--scope",
                "session",
                "--session",
                "session-123",
            ],
        )

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["_meta"]["schema_version"] == "headroom-knowledge-worker/v1"
        assert data["_meta"]["memory_count"] == 5
        assert any(node["type"] == "source" for node in data["nodes"].values())
        assert any(edge["type"] == "MENTIONED_IN" for edge in data["edges"])
        rendered = json.dumps(data)
        assert "Working on authentication feature" in rendered
        assert "User prefers TypeScript over JavaScript" not in rendered

    def test_export_context_format(self, runner: CliRunner, populated_db: str) -> None:
        """Context export emits a compact Markdown handoff."""
        result = runner.invoke(
            main,
            [
                "memory",
                "export",
                "--db-path",
                populated_db,
                "--format",
                "context",
                "--scope",
                "session",
                "--session",
                "session-123",
            ],
        )

        assert result.exit_code == 0
        assert "# Headroom Memory Context" in result.output
        assert "Working on authentication feature" in result.output

    @pytest.mark.parametrize(
        ("export_format", "suffix", "expected"),
        [
            ("knowledge-worker", ".json", "headroom-knowledge-worker/v1"),
            ("context", ".md", "# Headroom Memory Context"),
        ],
    )
    def test_scoped_export_writes_rendered_output_to_file(
        self,
        runner: CliRunner,
        populated_db: str,
        tmp_path: Path,
        export_format: str,
        suffix: str,
        expected: str,
    ) -> None:
        output_file = tmp_path / f"memory-export{suffix}"
        result = runner.invoke(
            main,
            [
                "memory",
                "export",
                "--db-path",
                populated_db,
                "--format",
                export_format,
                "--scope",
                "session",
                "--session",
                "session-123",
                "--output",
                str(output_file),
            ],
        )

        assert result.exit_code == 0, result.output
        assert expected in output_file.read_text(encoding="utf-8")

    def test_memory_audit_outputs_weak_claim_queue(
        self, runner: CliRunner, populated_db: str
    ) -> None:
        """Audit output includes provenance coverage and a review queue."""
        result = runner.invoke(
            main,
            ["memory", "audit", "--db-path", populated_db, "--scope", "global"],
        )

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["schema_version"] == "headroom-memory-audit/v1"
        assert data["provenance_coverage"]["node_coverage"] == 1.0
        assert data["ranked"]["weak_claim_queue"]

    def test_import_from_file(self, runner: CliRunner, temp_db: str, tmp_path: Path) -> None:
        """Import memories from file."""
        # Create import file
        import_data = [
            {
                "id": "imported-001",
                "content": "Imported memory",
                "user_id": "test-user",
                "importance": 0.8,
                "created_at": datetime.now().isoformat(),
                "valid_from": datetime.now().isoformat(),
            }
        ]
        import_file = tmp_path / "import.json"
        with open(import_file, "w") as f:
            json.dump(import_data, f)

        # Initialize empty db
        SQLiteMemoryStore(temp_db)

        result = runner.invoke(
            main,
            ["memory", "import", "--db-path", temp_db, str(import_file), "--force"],
        )
        assert result.exit_code == 0

        # Verify imported
        show_result = runner.invoke(main, ["memory", "show", "--db-path", temp_db, "imported-001"])
        assert "Imported memory" in show_result.output

    def test_export_import_roundtrip(
        self, runner: CliRunner, populated_db: str, tmp_path: Path
    ) -> None:
        """Export and import should be lossless."""
        export_file = tmp_path / "roundtrip.json"
        new_db = str(tmp_path / "new.db")

        # Export
        runner.invoke(
            main, ["memory", "export", "--db-path", populated_db, "--output", str(export_file)]
        )

        # Import to new db
        SQLiteMemoryStore(new_db)
        runner.invoke(main, ["memory", "import", "--db-path", new_db, str(export_file), "--force"])

        # Compare stats
        orig_stats = runner.invoke(main, ["memory", "stats", "--db-path", populated_db])
        new_stats = runner.invoke(main, ["memory", "stats", "--db-path", new_db])

        # Should have same count
        assert "6" in orig_stats.output
        assert "6" in new_stats.output


class TestMemoryHelp:
    """Tests for help output."""

    def test_memory_help(self, runner: CliRunner) -> None:
        """Memory group shows help."""
        result = runner.invoke(main, ["memory", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "show" in result.output
        assert "stats" in result.output
        assert "edit" in result.output
        assert "delete" in result.output
        assert "prune" in result.output
        assert "purge" in result.output
        assert "export" in result.output
        assert "import" in result.output

    def test_list_help(self, runner: CliRunner) -> None:
        """List command shows help."""
        result = runner.invoke(main, ["memory", "list", "--help"])
        assert result.exit_code == 0
        assert "--limit" in result.output
        assert "--scope" in result.output
        assert "--since" in result.output


class TestMemoryProvenanceMetadata:
    """Tests for save-time provenance enrichment."""

    def test_enrich_memory_metadata_adds_audit_fields(self) -> None:
        """Promotion metadata includes provenance, confidence, reason, and scope."""
        metadata = _enrich_memory_metadata(
            {"created_via": "tool_call"},
            content="User prefers local-first memory.",
            importance=0.9,
            user_id="alice",
            session_id="session-1",
            agent_id=None,
            turn_id=None,
        )

        assert metadata["source_text"] == "User prefers local-first memory."
        assert metadata["source_excerpt"] == "User prefers local-first memory."
        assert metadata["confidence"] == "high"
        assert metadata["promotion_rationale"]
        assert metadata["promoted_at_utc"]
        assert metadata["scope"] == "session"
        assert metadata["scope_details"]["session_id"] == "session-1"

    def test_enrich_memory_metadata_preserves_caller_provenance(self) -> None:
        """Caller-supplied richer provenance should not be overwritten."""
        metadata = _enrich_memory_metadata(
            {
                "source_text": "Full source note",
                "source_excerpt": "literal quote",
                "confidence": "low",
                "promotion_rationale": "manual review",
            },
            content="Derived claim",
            importance=0.9,
            user_id="alice",
            session_id=None,
            agent_id=None,
            turn_id=None,
        )

        assert metadata["source_text"] == "Full source note"
        assert metadata["source_excerpt"] == "literal quote"
        assert metadata["confidence"] == "low"
        assert metadata["promotion_rationale"] == "manual review"

    def test_enrich_and_export_preserve_source_chunk_over_derived_claim(
        self, runner: CliRunner, temp_db: str
    ) -> None:
        """source_chunk='Launch undecided' with content='Launch Friday' preserves source_chunk."""
        metadata = _enrich_memory_metadata(
            {"source_chunk": "Launch undecided"},
            content="Launch Friday",
            importance=0.8,
            user_id="alice",
            session_id="session-1",
            agent_id=None,
            turn_id=None,
        )
        assert metadata["source_text"] == "Launch undecided"
        assert metadata["source_excerpt"] == "Launch undecided"

        store = SQLiteMemoryStore(temp_db)
        asyncio.run(
            store.save(
                Memory(
                    id="mem-launch",
                    content="Launch Friday",
                    user_id="alice",
                    session_id="session-1",
                    importance=0.8,
                    metadata={
                        "source_chunk": "Launch undecided",
                        "source_text": "Launch Friday",
                        "source_excerpt": "Launch Friday",
                    },
                )
            )
        )

        result = runner.invoke(
            main,
            [
                "memory",
                "export",
                "--db-path",
                temp_db,
                "--format",
                "knowledge-worker",
                "--scope",
                "global",
            ],
        )
        assert result.exit_code == 0, result.output
        graph = json.loads(result.output)
        source_node = graph["nodes"]["source:headroom-memory-mem-launch"]
        assert source_node["body"] == "Launch undecided"
        mentioned_edges = [edge for edge in graph["edges"] if edge["type"] == "MENTIONED_IN"]
        assert len(mentioned_edges) == 1
        assert mentioned_edges[0]["excerpt"] == "Launch undecided"

    def test_pre_extracted_facts_do_not_cite_derived_fact_as_excerpt(self, tmp_path: Path) -> None:
        """Pre-extracted facts preserve parent passage or omit synthetic self-quotes."""
        from types import SimpleNamespace

        from headroom.memory.backends.local import LocalBackend
        from headroom.memory.knowledge_worker import _memory_excerpt

        def make_backend(db_file: Path) -> LocalBackend:
            store = SQLiteMemoryStore(str(db_file))
            backend = LocalBackend()
            backend._initialized = True

            async def add_to_sqlite(**kwargs: object) -> Memory:
                mem = Memory(
                    id=f"fact-{kwargs.get('user_id')}",
                    content=str(kwargs["content"]),
                    user_id=str(kwargs["user_id"]),
                    session_id=kwargs.get("session_id"),  # type: ignore[arg-type]
                    agent_id=kwargs.get("agent_id"),  # type: ignore[arg-type]
                    turn_id=kwargs.get("turn_id"),  # type: ignore[arg-type]
                    importance=float(kwargs.get("importance", 0.5)),  # type: ignore[arg-type]
                    entity_refs=list(kwargs.get("entity_refs") or []),  # type: ignore[arg-type]
                    metadata=dict(kwargs.get("metadata") or {}),  # type: ignore[arg-type]
                )
                await store.save(mem)
                return mem

            backend._hierarchical_memory = SimpleNamespace(add=add_to_sqlite)  # type: ignore[assignment]
            backend._graph = SimpleNamespace()  # type: ignore[assignment]
            return backend

        backend_passage = make_backend(tmp_path / "passage.db")
        saved_from_passage = asyncio.run(
            backend_passage.save_memory(
                content="Meeting transcript: Launch undecided until Q3.",
                user_id="alice",
                importance=0.8,
                facts=["Launch Friday"],
            )
        )
        assert (
            saved_from_passage.metadata["source_text"]
            == "Meeting transcript: Launch undecided until Q3."
        )
        assert (
            saved_from_passage.metadata["source_excerpt"]
            == "Meeting transcript: Launch undecided until Q3."
        )
        assert (
            _memory_excerpt(saved_from_passage) == "Meeting transcript: Launch undecided until Q3."
        )

        backend_synthetic = make_backend(tmp_path / "synthetic.db")
        saved_synthetic = asyncio.run(
            backend_synthetic.save_memory(
                content="Launch Friday",
                user_id="alice",
                importance=0.8,
                facts=["Launch Friday"],
            )
        )
        assert saved_synthetic.metadata["source_excerpt"] == ""
        assert saved_synthetic.metadata.get("source_is_synthetic") is True
        assert _memory_excerpt(saved_synthetic) == ""


class TestKnowledgeWorkerScopeAndGraphBoundaries:
    """Focused regression tests for PR #1586 export/audit review findings."""

    def test_load_memories_uses_unbounded_filter(self, temp_db: str) -> None:
        """load_memories must not cap queries at 100,000 before scope filtering."""
        from headroom.memory.knowledge_worker import load_memories
        from headroom.memory.ports import MemoryFilter

        store = SQLiteMemoryStore(temp_db)
        captured_filters: list[MemoryFilter] = []
        original_query = store.query

        async def spy_query(filter_obj: MemoryFilter) -> list[Memory]:
            captured_filters.append(filter_obj)
            return await original_query(filter_obj)

        store.query = spy_query  # type: ignore[method-assign]
        asyncio.run(load_memories(store, include_superseded=False))

        assert len(captured_filters) == 1
        assert captured_filters[0].limit is None
        assert captured_filters[0].include_superseded is False

    def test_project_scope_rejects_mixed_workspaces_and_supports_per_project_db(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """--scope project without --project rejects shared DBs but works on per-project DB paths."""
        shared_db = str(tmp_path / "shared_memory.db")
        store = SQLiteMemoryStore(shared_db)
        asyncio.run(
            store.save(
                Memory(
                    id="alpha-mem",
                    content="Alpha workspace architecture note",
                    user_id="alice",
                    importance=0.8,
                    metadata={"storage_mode": "project", "workspace_key": "alpha"},
                )
            )
        )
        asyncio.run(
            store.save(
                Memory(
                    id="beta-mem",
                    content="Beta workspace secret roadmap",
                    user_id="alice",
                    importance=0.8,
                    metadata={"storage_mode": "project", "workspace_key": "beta"},
                )
            )
        )

        # Shared DB without --project must fail instead of mixing alpha + beta
        unscoped_project = runner.invoke(
            main,
            [
                "memory",
                "export",
                "--db-path",
                shared_db,
                "--format",
                "knowledge-worker",
                "--scope",
                "project",
            ],
        )
        assert unscoped_project.exit_code != 0
        assert "project scope requires --project" in unscoped_project.output

        # Shared DB with explicit --project alpha exports only alpha
        explicit_alpha = runner.invoke(
            main,
            [
                "memory",
                "export",
                "--db-path",
                shared_db,
                "--format",
                "knowledge-worker",
                "--scope",
                "project",
                "--project",
                "alpha",
            ],
        )
        assert explicit_alpha.exit_code == 0, explicit_alpha.output
        alpha_graph = json.loads(explicit_alpha.output)
        assert alpha_graph["_meta"]["memory_count"] == 1
        assert "Alpha workspace architecture note" in explicit_alpha.output
        assert "Beta workspace secret roadmap" not in explicit_alpha.output

        # Per-project DB path (.../projects/alpha/memory.db) infers project key without --project
        per_project_dir = tmp_path / "projects" / "alpha"
        per_project_dir.mkdir(parents=True)
        per_project_db = str(per_project_dir / "memory.db")
        per_project_store = SQLiteMemoryStore(per_project_db)
        asyncio.run(
            per_project_store.save(
                Memory(
                    id="per-proj-mem",
                    content="Per-project DB memory",
                    user_id="alice",
                    importance=0.8,
                )
            )
        )
        inferred_result = runner.invoke(
            main,
            [
                "memory",
                "export",
                "--db-path",
                per_project_db,
                "--format",
                "knowledge-worker",
                "--scope",
                "project",
            ],
        )
        assert inferred_result.exit_code == 0, inferred_result.output
        inferred_graph = json.loads(inferred_result.output)
        assert inferred_graph["_meta"]["memory_count"] == 1

    def test_session_scope_requires_session_and_isolates_graph_entities(
        self, runner: CliRunner, temp_db: str
    ) -> None:
        """--scope session requires --session and excludes other sessions' graph entities/edges."""
        from headroom.cli.memory import derive_graph_db_path
        from headroom.memory.adapters.graph_models import Entity, Relationship
        from headroom.memory.adapters.sqlite_graph import SQLiteGraphStore

        store = SQLiteMemoryStore(temp_db)
        graph_store = SQLiteGraphStore(derive_graph_db_path(temp_db))

        asyncio.run(
            store.save(
                Memory(
                    id="sess-a-mem",
                    content="Session A discusses Project Orion",
                    user_id="alice",
                    session_id="session-a",
                    importance=0.8,
                )
            )
        )
        asyncio.run(
            store.save(
                Memory(
                    id="sess-b-mem",
                    content="Session B discusses Project Nova",
                    user_id="alice",
                    session_id="session-b",
                    importance=0.8,
                )
            )
        )
        asyncio.run(
            graph_store.add_entity(
                Entity(
                    id="ent-orion",
                    user_id="alice",
                    name="Project Orion",
                    entity_type="project",
                    metadata={"source_memory_id": "sess-a-mem"},
                )
            )
        )
        asyncio.run(
            graph_store.add_entity(
                Entity(
                    id="ent-nova",
                    user_id="alice",
                    name="Project Nova",
                    entity_type="project",
                    metadata={"source_memory_id": "sess-b-mem"},
                )
            )
        )
        asyncio.run(
            graph_store.add_relationship(
                Relationship(
                    id="rel-cross",
                    user_id="alice",
                    source_id="ent-orion",
                    target_id="ent-nova",
                    relation_type="depends_on",
                    metadata={"source_memory_id": "sess-b-mem"},
                )
            )
        )

        missing_session = runner.invoke(
            main,
            [
                "memory",
                "export",
                "--db-path",
                temp_db,
                "--format",
                "knowledge-worker",
                "--scope",
                "session",
            ],
        )
        assert missing_session.exit_code != 0
        assert "session scope requires --session" in missing_session.output

        scoped_a = runner.invoke(
            main,
            [
                "memory",
                "export",
                "--db-path",
                temp_db,
                "--format",
                "knowledge-worker",
                "--scope",
                "session",
                "--session",
                "session-a",
            ],
        )
        assert scoped_a.exit_code == 0, scoped_a.output
        graph_a = json.loads(scoped_a.output)
        labels = {node["label"] for node in graph_a["nodes"].values()}
        assert "Project Orion" in labels
        assert "Project Nova" not in labels
        assert all(edge["type"] != "DEPENDS_ON" for edge in graph_a["edges"])

    def test_relationships_exclude_superseded_and_sourceless_and_about_reuses_typed_entities(
        self, runner: CliRunner, temp_db: str
    ) -> None:
        """Superseded/sourceless relationships are excluded and ABOUT edges link typed graph entities."""
        from headroom.cli.memory import derive_graph_db_path
        from headroom.memory.adapters.graph_models import Entity, Relationship
        from headroom.memory.adapters.sqlite_graph import SQLiteGraphStore

        store = SQLiteMemoryStore(temp_db)
        graph_store = SQLiteGraphStore(derive_graph_db_path(temp_db))

        asyncio.run(
            store.save(
                Memory(
                    id="mem-superseded",
                    content="Old claim: Alice reports to Carol",
                    user_id="alice",
                    importance=0.7,
                    superseded_by="mem-active",
                    entity_refs=["Alice", "Carol"],
                )
            )
        )
        asyncio.run(
            store.save(
                Memory(
                    id="mem-active",
                    content="Active claim: Alice leads Project Atlas",
                    user_id="alice",
                    importance=0.9,
                    entity_refs=["Alice", "Project Atlas"],
                )
            )
        )

        asyncio.run(
            graph_store.add_entity(
                Entity(
                    id="ent-alice",
                    user_id="alice",
                    name="Alice",
                    entity_type="person",
                    metadata={"source_memory_id": "mem-active"},
                )
            )
        )
        asyncio.run(
            graph_store.add_entity(
                Entity(
                    id="ent-atlas",
                    user_id="alice",
                    name="Project Atlas",
                    entity_type="project",
                    metadata={"source_memory_id": "mem-active"},
                )
            )
        )
        asyncio.run(
            graph_store.add_entity(
                Entity(
                    id="ent-carol",
                    user_id="alice",
                    name="Carol",
                    entity_type="person",
                    metadata={"source_memory_id": "mem-superseded"},
                )
            )
        )
        asyncio.run(
            graph_store.add_relationship(
                Relationship(
                    id="rel-active",
                    user_id="alice",
                    source_id="ent-alice",
                    target_id="ent-atlas",
                    relation_type="leads",
                    metadata={"source_memory_id": "mem-active"},
                )
            )
        )
        asyncio.run(
            graph_store.add_relationship(
                Relationship(
                    id="rel-superseded",
                    user_id="alice",
                    source_id="ent-alice",
                    target_id="ent-carol",
                    relation_type="reports_to",
                    metadata={"source_memory_id": "mem-superseded"},
                )
            )
        )
        asyncio.run(
            graph_store.add_relationship(
                Relationship(
                    id="rel-sourceless",
                    user_id="alice",
                    source_id="ent-alice",
                    target_id="ent-atlas",
                    relation_type="unverified_link",
                    metadata={},
                )
            )
        )

        active_export = runner.invoke(
            main,
            [
                "memory",
                "export",
                "--db-path",
                temp_db,
                "--format",
                "knowledge-worker",
                "--scope",
                "global",
            ],
        )
        assert active_export.exit_code == 0, active_export.output
        graph = json.loads(active_export.output)

        # Typed entity IDs are used and no duplicate topic:* nodes are created for Alice / Project Atlas
        assert "person:alice-ent-alice" in graph["nodes"]
        assert "project:project-atlas-ent-atlas" in graph["nodes"]
        assert not any(k.startswith("topic:alice") for k in graph["nodes"])
        assert not any(k.startswith("topic:project-atlas") for k in graph["nodes"])
        assert not any(k.startswith("person:carol") for k in graph["nodes"])

        about_targets = {edge["dst"] for edge in graph["edges"] if edge["type"] == "ABOUT"}
        assert about_targets == {"person:alice-ent-alice", "project:project-atlas-ent-atlas"}

        rel_types = {edge["type"] for edge in graph["edges"]}
        assert "LEADS" in rel_types
        assert "REPORTS_TO" not in rel_types
        assert "UNVERIFIED_LINK" not in rel_types
        assert all(bool(edge.get("source_id")) for edge in graph["edges"])

        # With --include-superseded, superseded relationship is included, sourceless still excluded
        superseded_export = runner.invoke(
            main,
            [
                "memory",
                "export",
                "--db-path",
                temp_db,
                "--format",
                "knowledge-worker",
                "--scope",
                "global",
                "--include-superseded",
            ],
        )
        assert superseded_export.exit_code == 0, superseded_export.output
        full_graph = json.loads(superseded_export.output)
        full_rel_types = {edge["type"] for edge in full_graph["edges"]}
        assert "REPORTS_TO" in full_rel_types
        assert "UNVERIFIED_LINK" not in full_rel_types
        assert all(bool(edge.get("source_id")) for edge in full_graph["edges"])

    def test_long_ids_with_shared_32_char_prefix_do_not_collide(
        self, runner: CliRunner, temp_db: str
    ) -> None:
        """Distinct IDs longer than 32 characters with a shared prefix produce distinct nodes."""
        store = SQLiteMemoryStore(temp_db)
        shared_prefix = "shared-memory-identifier-prefix-000000"
        asyncio.run(
            store.save(
                Memory(
                    id=f"{shared_prefix}-first-record",
                    content="First long-ID memory claim",
                    user_id="alice",
                    importance=0.8,
                )
            )
        )
        asyncio.run(
            store.save(
                Memory(
                    id=f"{shared_prefix}-second-record",
                    content="Second long-ID memory claim",
                    user_id="alice",
                    importance=0.8,
                )
            )
        )

        result = runner.invoke(
            main,
            [
                "memory",
                "export",
                "--db-path",
                temp_db,
                "--format",
                "knowledge-worker",
                "--scope",
                "global",
            ],
        )
        assert result.exit_code == 0, result.output
        graph = json.loads(result.output)
        source_nodes = [
            node_id for node_id, node in graph["nodes"].items() if node["type"] == "source"
        ]
        claim_nodes = [
            node_id for node_id, node in graph["nodes"].items() if node["type"] != "source"
        ]
        assert len(source_nodes) == 2
        assert len(claim_nodes) == 2
