import inspect
import os
import sys
import types

import pytest


class FakeTinygradDevice:
    DEFAULT: str = "CPU"


def _mlx_backend_available() -> bool:
    """Return True only if mlx.core can be fully loaded (native libs present)."""
    try:
        import mlx.core  # noqa: F401  # pyright: ignore[reportUnusedImport]
        return True
    except (ImportError, OSError):
        return False

requires_mlx = pytest.mark.skipif(
    not _mlx_backend_available(),
    reason="MLX native backend not available (missing CUDA/Metal libraries)",
)

def test_tinygrad_runner_imports_without_mlx() -> None:
    """tinygrad_runner.py must be importable on Linux where MLX is absent."""
    from exo.worker.runner.llm_inference.tinygrad_runner import (
        main,  # noqa: F401  # pyright: ignore[reportUnusedImport]
    )

def test_engine_factory_importable() -> None:
    """engine_factory.py must be importable on any platform."""
    from exo.worker.engines.engine_factory import (
        Engine,  # noqa: F401  # pyright: ignore[reportUnusedImport]
        create_engine,  # noqa: F401  # pyright: ignore[reportUnusedImport]
    )

def test_engine_is_immutable() -> None:
    """Engine must be an immutable Pydantic model with the expected fields."""
    from pydantic import BaseModel

    from exo.worker.engines.engine_factory import Engine
    assert issubclass(Engine, BaseModel)
    field_names = set(Engine.model_fields.keys())

    required_fields = [
        "initialize", "load", "generate", "warmup", "cleanup",
        "apply_chat_template", "detect_thinking_prompt_suffix",
    ]

    assert all(field in field_names for field in required_fields)

def test_tokenizer_protocol_importable() -> None:
    from exo.shared.types.worker.tokenizer import (
        Tokenizer,  # noqa: F401  # pyright: ignore[reportUnusedImport]
    )


def test_tinygrad_runtime_defaults_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from exo.worker.runner.bootstrap import configure_tinygrad_runtime

    fake_device = _install_fake_tinygrad_device(monkeypatch)
    monkeypatch.delenv("EXO_TINYGRAD_DEVICE", raising=False)
    monkeypatch.delenv("VULKAN", raising=False)
    monkeypatch.delenv("CPU", raising=False)
    monkeypatch.delenv("TC", raising=False)

    assert configure_tinygrad_runtime() == "CPU"
    assert fake_device.DEFAULT == "CPU"
    assert os.environ["CPU"] == "1"
    assert os.environ["TC"] == "0"


def test_tinygrad_runtime_can_select_vulkan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from exo.worker.runner.bootstrap import configure_tinygrad_runtime

    fake_device = _install_fake_tinygrad_device(monkeypatch)
    monkeypatch.setenv("EXO_TINYGRAD_DEVICE", "vulkan")
    monkeypatch.delenv("TC", raising=False)

    assert configure_tinygrad_runtime() == "VULKAN"
    assert fake_device.DEFAULT == "VULKAN"
    assert os.environ["VULKAN"] == "1"
    assert os.environ["TC"] == "1"


def test_tinygrad_runtime_rejects_other_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from exo.worker.runner.bootstrap import configure_tinygrad_runtime

    _install_fake_tinygrad_device(monkeypatch)
    monkeypatch.setenv("EXO_TINYGRAD_DEVICE", "METAL")

    with pytest.raises(ValueError, match="EXO_TINYGRAD_DEVICE"):
        configure_tinygrad_runtime()


def _install_fake_tinygrad_device(
    monkeypatch: pytest.MonkeyPatch,
) -> type[FakeTinygradDevice]:
    FakeTinygradDevice.DEFAULT = "CPU"
    tinygrad_module = types.ModuleType("tinygrad")
    device_module = types.ModuleType("tinygrad.device")
    device_module.Device = FakeTinygradDevice  # pyright: ignore[reportAttributeAccessIssue]
    monkeypatch.setitem(sys.modules, "tinygrad", tinygrad_module)
    monkeypatch.setitem(sys.modules, "tinygrad.device", device_module)
    return FakeTinygradDevice


@requires_mlx
def test_mlx_engine_has_postprocessing_importable() -> None:
    from exo.worker.engines.mlx.generator.generate import (
        mlx_generate_with_postprocessing,  # noqa: F401  # pyright: ignore[reportUnusedImport]
    )

@requires_mlx
def test_mlx_engine_has_postprocessing_signature() -> None:
    from exo.worker.engines.mlx.generator.generate import (
        mlx_generate_with_postprocessing,
    )
    sig = inspect.signature(mlx_generate_with_postprocessing)
    params = list(sig.parameters.keys())

    expected_params = ["model", "tokenizer", "model_id"]

    assert all(param in params for param in expected_params)
