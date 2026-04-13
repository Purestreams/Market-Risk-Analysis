from __future__ import annotations

from dataclasses import dataclass
import sys
import types
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class NPUExecutionPlan:
    library_imported: bool
    torch_device_available: bool
    using_npu: bool
    backend: str
    note: str


def _patch_neural_compressor_namespace() -> str:
    class WeightOnlyLinear(nn.Module):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__()

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # pragma: no cover - shim only
            return inputs

    class TuningCriterion:
        def __init__(self, timeout: int = 0) -> None:
            self.timeout = timeout

    class PostTrainingQuantConfig:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

    def fit(*args: Any, **kwargs: Any) -> torch.nn.Module:  # pragma: no cover - quantized path unused in this project
        raise RuntimeError("Quantized Intel NPU paths are not enabled in this project shim.")

    adaptor_module = types.ModuleType("neural_compressor.adaptor")
    adaptor_module.__path__ = []
    sys.modules["neural_compressor.adaptor"] = adaptor_module

    torch_utils_module = types.ModuleType("neural_compressor.adaptor.torch_utils")
    torch_utils_module.__path__ = []
    sys.modules["neural_compressor.adaptor.torch_utils"] = torch_utils_module

    wrapper_module = types.ModuleType("neural_compressor.adaptor.torch_utils.model_wrapper")
    wrapper_module.WeightOnlyLinear = WeightOnlyLinear
    sys.modules["neural_compressor.adaptor.torch_utils.model_wrapper"] = wrapper_module

    config_module = types.ModuleType("neural_compressor.config")
    config_module.PostTrainingQuantConfig = PostTrainingQuantConfig
    config_module.TuningCriterion = TuningCriterion
    sys.modules["neural_compressor.config"] = config_module

    quantization_module = types.ModuleType("neural_compressor.quantization")
    quantization_module.fit = fit
    sys.modules["neural_compressor.quantization"] = quantization_module
    return "compatibility patch applied"


def get_npu_execution_plan() -> tuple[NPUExecutionPlan, Any | None, Any | None]:
    patch_note = _patch_neural_compressor_namespace()
    try:
        import intel_npu_acceleration_library as npu_lib
        from intel_npu_acceleration_library.compiler import CompilerConfig
    except Exception as exc:  # pragma: no cover - hardware/runtime dependent
        return (
            NPUExecutionPlan(
                library_imported=False,
                torch_device_available=False,
                using_npu=False,
                backend="cpu-fallback",
                note=f"{patch_note}; Intel NPU import failed: {exc}",
            ),
            None,
            None,
        )

    torch_device_available = False
    note = f"{patch_note}; Intel NPU library imported"
    try:
        torch.zeros(1).to("npu")
        torch_device_available = True
        note = f"{note}; torch npu device is available"
    except Exception as exc:  # pragma: no cover - hardware/runtime dependent
        note = f"{note}; torch npu device unavailable: {exc}"

    return (
        NPUExecutionPlan(
            library_imported=True,
            torch_device_available=torch_device_available,
            using_npu=torch_device_available,
            backend="intel-npu-acceleration-library",
            note=note,
        ),
        npu_lib,
        CompilerConfig,
    )


def select_training_device() -> tuple[torch.device, str]:
    plan, _, _ = get_npu_execution_plan()
    if plan.torch_device_available:
        try:
            return torch.device("npu"), plan.note
        except Exception as exc:  # pragma: no cover - hardware/runtime dependent
            return torch.device("cpu"), f"{plan.note}; device selection fell back to CPU: {exc}"
    return torch.device("cpu"), plan.note


def maybe_compile_for_npu(module: torch.nn.Module, module_name: str) -> tuple[torch.nn.Module, NPUExecutionPlan]:
    plan, npu_lib, compiler_config = get_npu_execution_plan()
    if not plan.library_imported or npu_lib is None or compiler_config is None:
        return module, plan

    try:
        compiled_module = npu_lib.compile(module.eval(), compiler_config(dtype=torch.float16, training=False))
        return (
            compiled_module,
            NPUExecutionPlan(
                library_imported=True,
                torch_device_available=plan.torch_device_available,
                using_npu=True,
                backend=plan.backend,
                note=f"{plan.note}; compiled {module_name} for NPU inference",
            ),
        )
    except Exception as exc:  # pragma: no cover - hardware/runtime dependent
        return (
            module,
            NPUExecutionPlan(
                library_imported=plan.library_imported,
                torch_device_available=plan.torch_device_available,
                using_npu=False,
                backend="cpu-fallback",
                note=f"{plan.note}; compile for {module_name} failed: {exc}",
            ),
        )