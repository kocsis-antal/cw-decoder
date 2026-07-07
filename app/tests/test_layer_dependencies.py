from __future__ import annotations

import ast
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src" / "cw"


def _cw_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("cw."):
            imports.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("cw."):
                    imports.add(alias.name)
    return imports


def _layer_files(layer: str) -> list[Path]:
    return [path for path in (SRC / layer).rglob("*.py") if "__pycache__" not in path.parts]


def _assert_no_forbidden_imports(layer: str, forbidden_prefixes: tuple[str, ...]) -> None:
    violations: list[str] = []
    for path in _layer_files(layer):
        for imported in sorted(_cw_imports(path)):
            if imported.startswith(forbidden_prefixes):
                violations.append(f"{path.relative_to(SRC)} -> {imported}")
    assert not violations, "Forbidden layer imports:\n" + "\n".join(violations)


def test_no_standalone_events_or_compatibility_layers_left() -> None:
    assert not (SRC / "events").exists()
    assert not (SRC / "receiving" / "events.py").exists()
    for facade in (
        "nextgen.py",
        "nextgen_stream.py",
        "stream_models.py",
        "stream_events.py",
        "output_events.py",
        "stream_sources.py",
        "event_view.py",
        "cli_app.py",
        "cli_stream.py",
    ):
        assert not (SRC / facade).exists(), f"legacy facade still exists: {facade}"
    assert not (SRC / "io" / "sources.py").exists()


def test_io_layer_has_no_runtime_dependencies() -> None:
    _assert_no_forbidden_imports(
        "io",
        ("cw.receiving", "cw.signal", "cw.decoder", "cw.selection", "cw.ui", "cw.app", "cw.tools"),
    )


def test_receiving_stops_before_signal_and_decoder_layers() -> None:
    _assert_no_forbidden_imports(
        "receiving",
        ("cw.signal", "cw.decoder", "cw.selection", "cw.ui", "cw.app", "cw.tools"),
    )


def test_signal_layer_depends_only_on_receiving_contracts() -> None:
    _assert_no_forbidden_imports(
        "signal",
        ("cw.decoder", "cw.selection", "cw.ui", "cw.app", "cw.tools"),
    )


def test_decoder_layer_depends_only_on_signal_or_inner_layers() -> None:
    _assert_no_forbidden_imports(
        "decoder",
        ("cw.io", "cw.receiving", "cw.selection", "cw.ui", "cw.app", "cw.tools"),
    )


def test_selection_layer_does_not_reach_into_io_or_ui_or_app() -> None:
    _assert_no_forbidden_imports(
        "selection",
        ("cw.io", "cw.receiving", "cw.signal", "cw.ui", "cw.app", "cw.tools"),
    )


def test_ui_reads_only_application_channel_output() -> None:
    allowed = {"cw.app.channel_output"}
    violations: list[str] = []
    for path in _layer_files("ui"):
        for imported in sorted(_cw_imports(path)):
            if imported in allowed:
                continue
            if imported.startswith("cw."):
                violations.append(f"{path.relative_to(SRC)} -> {imported}")
    assert not violations, "UI imported non-output internals:\n" + "\n".join(violations)


def test_runtime_layer_names_are_functional_not_nextgen_or_legacy_live() -> None:
    violations: list[str] = []
    forbidden_fragments = ("Nextgen", "nextgen", "LiveStreamProcessor", "StreamingConfig")
    for layer in ("io", "receiving", "signal", "decoder", "selection", "app", "ui"):
        for path in _layer_files(layer):
            relative = path.relative_to(SRC)
            text = path.read_text(encoding="utf-8")
            if "nextgen" in path.name.lower():
                violations.append(f"{relative}: filename contains nextgen")
            for fragment in forbidden_fragments:
                if fragment in text:
                    violations.append(f"{relative}: runtime layer mentions {fragment}")
    assert not violations, "Non-functional legacy naming leaked into runtime layers:\n" + "\n".join(violations)


def test_decoder_runtime_layer_has_no_legacy_carrier_window_modules() -> None:
    legacy_files = {
        "base.py",
        "carrier_decode.py",
        "carrier_detection.py",
        "carrier_search.py",
        "models.py",
        "report.py",
        "stream_decode.py",
        "stream_models.py",
        "threshold_decoder.py",
        "soft_decoder.py",
        "symbol_hmm_decoder.py",
        "character_hmm_decoder.py",
        "lattice_decoder.py",
        "session_grouping.py",
        "signal_analysis.py",
        "quality.py",
    }
    present = {path.name for path in (SRC / "decoder").glob("*.py")}
    assert not (present & legacy_files)
