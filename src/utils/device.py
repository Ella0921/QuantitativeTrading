"""
Device configuration utility.

Handles the TF >= 2.11 Windows GPU limitation:
  - On Windows: uses tensorflow-directml-plugin if available, else CPU
  - On Linux/Mac: standard CUDA GPU detection

Call configure_gpu() once at the start of any training script.

Interview note: this is a common pain point in ML engineering —
knowing how to handle cross-platform GPU setup shows practical experience.
"""

from __future__ import annotations

import os
import platform
import sys


def configure_gpu(memory_limit_mb: int | None = None, verbose: bool = True) -> str:
    """
    Configure TensorFlow to use available GPU.

    Returns a string describing what device will be used:
    "directml", "cuda", or "cpu"

    Args:
        memory_limit_mb: if set, cap GPU memory usage (useful on shared machines)
        verbose: print device info
    """
    import tensorflow as tf

    system = platform.system()

    # ── Windows: try DirectML first ───────────────────────────────────────────
    if system == "Windows":
        try:
            import tensorflow_directml_plugin  # noqa: F401
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                if memory_limit_mb:
                    for gpu in gpus:
                        tf.config.set_logical_device_configuration(
                            gpu,
                            [tf.config.LogicalDeviceConfiguration(
                                memory_limit=memory_limit_mb
                            )]
                        )
                if verbose:
                    print(f"[device] Using DirectML GPU: {[g.name for g in gpus]}")
                return "directml"
        except ImportError:
            pass

        if verbose:
            print(
                "[device] GPU not available on Windows with standard TF >= 2.11.\n"
                "         Options:\n"
                "         1. pip install tensorflow-directml-plugin  (recommended)\n"
                "         2. Use WSL2 with CUDA\n"
                "         3. Continue on CPU (slower but functional)\n"
                "         Using CPU for now."
            )
        return "cpu"

    # ── Linux / Mac: standard CUDA / Metal ────────────────────────────────────
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            # Allow memory growth — prevents TF from grabbing all GPU memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            if memory_limit_mb:
                for gpu in gpus:
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(
                            memory_limit=memory_limit_mb
                        )]
                    )
            if verbose:
                print(f"[device] Using CUDA GPU: {[g.name for g in gpus]}")
            return "cuda"
        except RuntimeError as e:
            if verbose:
                print(f"[device] GPU config error: {e} — falling back to CPU")
            return "cpu"

    # ── Apple Silicon (MPS) ───────────────────────────────────────────────────
    if sys.platform == "darwin":
        try:
            mps = tf.config.list_physical_devices("GPU")
            if mps:
                if verbose:
                    print(f"[device] Using Apple MPS: {[g.name for g in mps]}")
                return "mps"
        except Exception:
            pass

    if verbose:
        print("[device] No GPU found — using CPU")
    return "cpu"


def device_summary() -> dict:
    """Return a dict summarising available compute devices."""
    import tensorflow as tf
    return {
        "platform":  platform.system(),
        "python":    sys.version.split()[0],
        "tf_version": tf.__version__,
        "cpus":      len(tf.config.list_physical_devices("CPU")),
        "gpus":      len(tf.config.list_physical_devices("GPU")),
        "gpu_names": [g.name for g in tf.config.list_physical_devices("GPU")],
    }
