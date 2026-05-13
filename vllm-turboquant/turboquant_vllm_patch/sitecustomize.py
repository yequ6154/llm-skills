"""Auto-install TurboQuant hooks for `vllm serve`.

This file is imported automatically by Python when it is on `PYTHONPATH`.
"""

from __future__ import annotations

import importlib
import os
from itertools import islice
from typing import Any


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


TQ_ENABLE = _env_bool("TQ_ENABLE", True)
TQ_KEY_BITS = _env_int("TQ_KEY_BITS", 3)
TQ_VALUE_BITS = _env_int("TQ_VALUE_BITS", 4)
TQ_BUFFER_SIZE = _env_int("TQ_BUFFER_SIZE", 128)
TQ_DIAG_VERBOSE = _env_bool("TQ_DIAG_VERBOSE", False)
TQ_MAX_INSTALL_ATTEMPTS = _env_int("TQ_MAX_INSTALL_ATTEMPTS", 6)


def _safe_log(msg: str) -> None:
    print(f"[turboquant-autoinstall] {msg}", flush=True)


def _diag_log(msg: str) -> None:
    if TQ_DIAG_VERBOSE:
        _safe_log(msg)


def _safe_type_name(obj: Any) -> str:
    t = type(obj)
    return f"{t.__module__}.{t.__name__}"


def _summarize_named_modules(model_obj: Any, limit: int = 24) -> list[str]:
    named_modules = getattr(model_obj, "named_modules", None)
    if named_modules is None:
        return []
    summary: list[str] = []
    try:
        for name, module in islice(named_modules(), limit):
            summary.append(f"{name}:{_safe_type_name(module)}")
    except Exception as exc:
        summary.append(f"<named_modules failed: {exc!r}>")
    return summary


def _find_model_object(model_runner: Any) -> Any:
    candidates = ("model", "module", "runner", "model_executor")
    for attr in candidates:
        obj = getattr(model_runner, attr, None)
        if obj is not None:
            return obj
    return None


def _log_worker_diagnostics(worker: Any, model_runner: Any, stage: str) -> None:
    try:
        _diag_log(
            f"diag stage={stage}, worker={_safe_type_name(worker)}, "
            f"model_runner={_safe_type_name(model_runner)}"
        )
        runner_attrs = sorted(set(dir(model_runner)))
        interesting = [
            x
            for x in runner_attrs
            if any(k in x.lower() for k in ("model", "attn", "cache", "kv", "layer"))
        ]
        _diag_log(f"diag runner attrs(sample)={interesting[:40]}")

        model_obj = _find_model_object(model_runner)
        if model_obj is None:
            _diag_log("diag no model-like object found on model_runner")
            return

        _diag_log(f"diag model object={_safe_type_name(model_obj)}")
        modules = _summarize_named_modules(model_obj)
        if modules:
            _diag_log("diag named_modules(sample)=" + " | ".join(modules))
    except Exception as exc:
        _diag_log(f"diag collection failed: {exc!r}")


def _install_for_worker(worker: Any, stage: str) -> None:
    if getattr(worker, "_tq_hook_count", 0) > 0:
        return
    attempts = int(getattr(worker, "_tq_install_attempts", 0))
    if attempts >= TQ_MAX_INSTALL_ATTEMPTS:
        if attempts == TQ_MAX_INSTALL_ATTEMPTS:
            _diag_log(f"skip install after max attempts={TQ_MAX_INSTALL_ATTEMPTS}")
            worker._tq_install_attempts = attempts + 1
        return

    model_runner = getattr(worker, "model_runner", None)
    if model_runner is None:
        _diag_log(f"install deferred stage={stage}: worker has no model_runner yet")
        return

    worker._tq_install_attempts = attempts + 1
    _log_worker_diagnostics(worker, model_runner, stage)

    from turboquant.vllm_attn_backend import MODE_ACTIVE, install_turboquant_hooks

    try:
        hooks = install_turboquant_hooks(
            model_runner,
            key_bits=TQ_KEY_BITS,
            value_bits=TQ_VALUE_BITS,
            buffer_size=TQ_BUFFER_SIZE,
            mode=MODE_ACTIVE,
        )
    except Exception as exc:
        _safe_log(
            f"hook install exception at stage={stage}, "
            f"attempt={worker._tq_install_attempts}: {exc!r}"
        )
        return

    worker._tq_hook_count = len(hooks) if hooks is not None else 0
    _safe_log(
        "hooks installed: "
        f"count={worker._tq_hook_count}, key_bits={TQ_KEY_BITS}, "
        f"value_bits={TQ_VALUE_BITS}, buffer={TQ_BUFFER_SIZE}, "
        f"stage={stage}, attempt={worker._tq_install_attempts}"
    )
    if worker._tq_hook_count == 0:
        _diag_log("diag install returned zero hooks; keep trying at later patch points")


def _patch_method(module_name: str, class_name: str, method_name: str) -> None:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return

    cls = getattr(module, class_name, None)
    if cls is None:
        return

    method = getattr(cls, method_name, None)
    if method is None or getattr(method, "_tq_wrapped", False):
        return

    def _wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        out = method(self, *args, **kwargs)
        try:
            _install_for_worker(self, f"{module_name}.{class_name}.{method_name}")
        except Exception as exc:
            _safe_log(f"hook install skipped on {class_name}.{method_name}: {exc!r}")
        return out

    _wrapped._tq_wrapped = True  # type: ignore[attr-defined]
    setattr(cls, method_name, _wrapped)
    _diag_log(f"patched {module_name}.{class_name}.{method_name}")


def _apply_patches() -> None:
    if not TQ_ENABLE:
        _safe_log("disabled by TQ_ENABLE=0")
        return

    _diag_log(
        "config: "
        f"enable={TQ_ENABLE}, key_bits={TQ_KEY_BITS}, value_bits={TQ_VALUE_BITS}, "
        f"buffer={TQ_BUFFER_SIZE}, diag_verbose={TQ_DIAG_VERBOSE}, "
        f"max_attempts={TQ_MAX_INSTALL_ATTEMPTS}"
    )
    patch_points = [
        ("vllm.worker.worker", "Worker", "load_model"),
        ("vllm.worker.worker", "Worker", "init_device"),
        ("vllm.worker.worker", "Worker", "execute_model"),
        ("vllm.worker.gpu_worker", "GPUWorker", "load_model"),
        ("vllm.worker.gpu_worker", "GPUWorker", "init_device"),
        ("vllm.worker.gpu_worker", "GPUWorker", "execute_model"),
        ("vllm.v1.worker.worker", "Worker", "load_model"),
        ("vllm.v1.worker.worker", "Worker", "init_device"),
        ("vllm.v1.worker.worker", "Worker", "execute_model"),
        ("vllm.v1.worker.gpu_worker", "Worker", "load_model"),
        ("vllm.v1.worker.gpu_worker", "GPUWorker", "load_model"),
        ("vllm.v1.worker.gpu_worker", "Worker", "init_device"),
        ("vllm.v1.worker.gpu_worker", "GPUWorker", "init_device"),
        ("vllm.v1.worker.gpu_worker", "Worker", "execute_model"),
        ("vllm.v1.worker.gpu_worker", "GPUWorker", "execute_model"),
    ]
    for module_name, class_name, method_name in patch_points:
        _patch_method(module_name, class_name, method_name)


_apply_patches()
