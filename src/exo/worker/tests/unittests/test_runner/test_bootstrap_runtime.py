from types import SimpleNamespace

import pytest

from exo.worker.runner import bootstrap


def test_configure_mlx_metal_fast_synch_skips_linux(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bootstrap.sys, "platform", "linux")
    monkeypatch.delenv("MLX_METAL_FAST_SYNCH", raising=False)

    bootstrap._configure_mlx_metal_fast_synch()

    assert "MLX_METAL_FAST_SYNCH" not in bootstrap.os.environ


def test_configure_mlx_metal_fast_synch_sets_darwin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bootstrap.sys, "platform", "darwin")
    monkeypatch.setenv("EXO_FAST_SYNCH", "false")
    monkeypatch.delenv("MLX_METAL_FAST_SYNCH", raising=False)

    bootstrap._configure_mlx_metal_fast_synch()

    assert bootstrap.os.environ["MLX_METAL_FAST_SYNCH"] == "0"


def test_validate_mlx_text_runtime_accepts_compatible_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        bootstrap.importlib,
        "import_module",
        lambda module_name: SimpleNamespace(new_thread_local_stream=object())
        if module_name == "mlx.core"
        else None,
    )

    bootstrap._validate_mlx_text_runtime()


def test_validate_mlx_text_runtime_rejects_incompatible_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        bootstrap.importlib,
        "import_module",
        lambda module_name: SimpleNamespace()
        if module_name == "mlx.core"
        else None,
    )

    with pytest.raises(RuntimeError, match="new_thread_local_stream"):
        bootstrap._validate_mlx_text_runtime()
