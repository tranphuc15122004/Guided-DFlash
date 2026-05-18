from __future__ import annotations

import os
import platform
import socket
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

from loguru import logger

try:
    import torch
except Exception:  # pragma: no cover - torch is expected in runtime
    torch = None


def log_run_parameters(
    script_name: str,
    args: Any,
    *,
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    values = vars(args) if hasattr(args, "__dict__") else dict(args)

    logger.info("=" * 100)
    logger.info(f"[{script_name}] run configuration")
    logger.info(f"cwd={Path.cwd()}")
    logger.info(f"hostname={socket.gethostname()}")
    logger.info(f"python={sys.version.split()[0]}")
    logger.info(f"platform={platform.platform()}")

    if torch is not None:
        logger.info(
            f"torch={torch.__version__}, cuda={torch.version.cuda}, cuda_available={torch.cuda.is_available()}"
        )

    for key in (
        "PBS_JOBID",
        "PBS_O_WORKDIR",
        "SCHEME",
        "HF_HOME",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
    ):
        if key in os.environ:
            logger.info(f"env.{key}={os.environ.get(key)!r}")

    for key in sorted(values):
        logger.info(f"arg.{key}={values[key]!r}")

    if extra:
        for key in sorted(extra):
            logger.info(f"extra.{key}={extra[key]!r}")

    logger.info("=" * 100)
