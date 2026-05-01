import os
import resource
from typing import Final

import loguru

from exo.shared.types.events import Event, RunnerStatusUpdated
from exo.shared.types.tasks import Task, TaskId
from exo.shared.types.worker.instances import BoundInstance, TinygradInstance
from exo.shared.types.worker.runners import RunnerFailed
from exo.utils.channels import ClosedResourceError, MpReceiver, MpSender
from exo.worker.engines.base import Builder

logger: "loguru.Logger" = loguru.logger

TINYGRAD_DEVICE_ENV: Final = "EXO_TINYGRAD_DEVICE"
SUPPORTED_TINYGRAD_DEVICES: Final = frozenset({"CPU", "CL", "WEBGPU", "VULKAN"})


def configure_tinygrad_runtime() -> str:
    requested_device = os.environ.get(TINYGRAD_DEVICE_ENV)
    if requested_device is None:
        requested_device = "VULKAN" if os.environ.get("VULKAN") == "1" else "CPU"
    device = requested_device.strip().upper()
    if device not in SUPPORTED_TINYGRAD_DEVICES:
        supported = ", ".join(sorted(SUPPORTED_TINYGRAD_DEVICES))
        raise ValueError(
            f"{TINYGRAD_DEVICE_ENV} must be one of {supported}, got {requested_device!r}"
        )

    os.environ.setdefault("JIT", "1")
    os.environ.setdefault("BEAM", "2")
    os.environ.setdefault("TC", "0" if device == "CPU" else "1")
    os.environ[device] = "1"

    from tinygrad.device import Device

    Device.DEFAULT = device
    return device


def entrypoint(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event],
    task_receiver: MpReceiver[Task],
    cancel_receiver: MpReceiver[TaskId],
    _logger: "loguru.Logger",
) -> None:
    global logger
    logger = _logger

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(max(soft, 2048), hard), hard))

    is_tinygrad = isinstance(bound_instance.instance, TinygradInstance)

    if is_tinygrad:
        tinygrad_device = configure_tinygrad_runtime()
        logger.info(f"Tinygrad device: {tinygrad_device}")
    else:
        fast_synch_override = os.environ.get("EXO_FAST_SYNCH")
        if fast_synch_override == "false":
            os.environ["MLX_METAL_FAST_SYNCH"] = "0"
        else:
            os.environ["MLX_METAL_FAST_SYNCH"] = "1"

        logger.info(f"Fast synch flag: {os.environ['MLX_METAL_FAST_SYNCH']}")

    try:
        if is_tinygrad:
            from exo.worker.runner.llm_inference.tinygrad_runner import (
                main as tinygrad_main,
            )

            tinygrad_main(bound_instance, event_sender, task_receiver, cancel_receiver)
            return

        from exo.worker.runner.runner import Runner

        builder: Builder

        if bound_instance.is_image_model:
            from exo.worker.engines.image.builder import MfluxBuilder

            builder = MfluxBuilder(
                event_sender, cancel_receiver, bound_instance.bound_shard
            )
        else:
            from exo.worker.engines.mlx.patches import apply_mlx_patches

            apply_mlx_patches()

            from exo.worker.engines.mlx.builder import MlxBuilder

            # evil sharing of the event sender
            builder = MlxBuilder(
                model_id=bound_instance.bound_shard.model_card.model_id,
                event_sender=event_sender,
                cancel_receiver=cancel_receiver,
            )

        runner = Runner(bound_instance, builder, event_sender, task_receiver)
        runner.main()

    except ClosedResourceError:
        logger.warning("Runner communication closed unexpectedly")
    except Exception as e:
        logger.opt(exception=e).warning(
            f"Runner {bound_instance.bound_runner_id} crashed with critical exception {e}"
        )
        event_sender.send(
            RunnerStatusUpdated(
                runner_id=bound_instance.bound_runner_id,
                runner_status=RunnerFailed(error_message=str(e)),
            )
        )
    finally:
        try:
            event_sender.close()
            task_receiver.close()
        finally:
            event_sender.join()
            task_receiver.join()
            logger.info("bye from the runner")
