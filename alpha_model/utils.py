import os
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, Optional, Union

from datasets import DownloadConfig, load_dataset


def _reasoning_prompt(text: str) -> str:
    return (
        f"{text}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    )


def _livecodebench_prompt(doc: dict) -> str:
    system_prompt = (
        "You are an expert Python programmer. You will be given a question (problem specification) "
        "and will generate a correct Python program that matches the specification and passes all tests. "
        "You will NOT return anything except for the program"
    )
    question_block = f"### Question:\n{doc['question_content']}"
    if doc.get("starter_code"):
        format_message = "### Format: Use the following code structure:"
        code_block = f"```python\n{doc['starter_code']}\n```"
    else:
        format_message = "### Format: Write your code in the following format:"
        code_block = "```python\n# YOUR CODE HERE\n```"
    answer_footer = "### Answer: (use the provided format with backticks)"
    return f"{system_prompt}\n\n{question_block}\n\n{format_message}\n{code_block}\n\n{answer_footer}"


def _ensure_turns(value: Union[str, list, tuple]) -> list:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    raise TypeError(f"turns must be str/list/tuple, got {type(value)}")


DatasetFormatter = Callable[[dict], list]

DATASET_CONFIG: Dict[str, dict] = {
    # Math datasets
    "gsm8k": {
        "path": "openai/gsm8k",
        "name": "main",
        "default_split": "train",
        "formatter": lambda x: [_reasoning_prompt(x["question"])],
    },
    "math500": {
        "path": "HuggingFaceH4/MATH-500",
        "default_split": "test",
        "formatter": lambda x: [_reasoning_prompt(x["problem"])],
    },
    "math_full": {
        "path": "qwedsacf/competition_math",
        "default_split": "train",
        "formatter": lambda x: [_reasoning_prompt(x["problem"])],
    },
    "math_instruct": {
        "path": "TIGER-Lab/MathInstruct",
        "default_split": "train",
        "formatter": lambda x: [x["instruction"]],
    },
    "metamath": {
        "path": "meta-math/MetaMathQA",
        "default_split": "train",
        "formatter": lambda x: [x["query"]],
    },
    "magicoder": {
        "path": "ise-uiuc/Magicoder-Evol-Instruct-110K",
        "default_split": "train",
        "formatter": lambda x: [x["instruction"]],
    },
    "aime24": {
        "path": "HuggingFaceH4/aime_2024",
        "default_split": "train",
        "formatter": lambda x: [_reasoning_prompt(x["problem"])],
    },
    "aime25": {
        "path": "MathArena/aime_2025",
        "default_split": "train",
        "formatter": lambda x: [_reasoning_prompt(x["problem"])],
    },
    # Chat datasets
    "alpaca": {
        "path": "tatsu-lab/alpaca",
        "default_split": "train",
        "formatter": lambda x: [
            f"{x['instruction']}\n\nInput:\n{x['input']}" if x.get("input") else x["instruction"]
        ],
    },
    "mt-bench": {
        "path": "HuggingFaceH4/mt_bench_prompts",
        "default_split": "train",
        "formatter": lambda x: _ensure_turns(x["prompt"]),
    },
    # Coding datasets
    "humaneval": {
        "path": "openai/openai_humaneval",
        "default_split": "test",
        "formatter": lambda x: [
            "Write a solution to the following problem and make sure that it passes the tests:\n"
            f"```python\n{x['prompt']}\n```"
        ],
    },
    "mbpp": {
        "path": "google-research-datasets/mbpp",
        "name": "sanitized",
        "default_split": "train",
        "formatter": lambda x: [x["prompt"]],
    },
    "lbpp": {
        "path": "parquet",
        "data_files": {
            "train": "https://huggingface.co/datasets/CohereLabs/lbpp/resolve/main/python/train.parquet",
            "test": "https://huggingface.co/datasets/CohereLabs/lbpp/resolve/main/python/test.parquet",
        },
        "default_split": "train",
        "formatter": lambda x: [x["instruction"]],
    },
    "swe-bench": {
        "path": "princeton-nlp/SWE-bench_Lite",
        "default_split": "test",
        "formatter": lambda x: [
            f"Problem Statement:\n{x['problem_statement']}\nPlease fix the issue described above."
        ],
    },
    "livecodebench": {
        "path": "json",
        "data_files": {
            "test": [
                "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/test.jsonl",
                "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/test2.jsonl",
                "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/test3.jsonl",
                "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/test4.jsonl",
                "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/test5.jsonl",
                "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/test6.jsonl",
            ]
        },
        "default_split": "test",
        "formatter": lambda x: [_livecodebench_prompt(x)],
    },
}


AUTO_STREAM_DATASETS = {"math_instruct", "metamath", "magicoder"}

LOCAL_MIRROR_FILENAMES = {
    "math_instruct": ["MathInstruct.json", "MathInstruct.jsonl"],
    "metamath": ["MetaMathQA-395K.json", "MetaMathQA.json", "MetaMathQA-395K.jsonl"],
    "magicoder": ["Magicoder-Evol-Instruct-110K.json", "Magicoder-Evol-Instruct-110K.jsonl"],
}

HF_OFFLINE_ENV_VARS = (
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_DATASETS_OFFLINE",
    "HF_DATASETS_OFFLINE_MODE",
)


def _is_truthy_env(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _has_offline_env_enabled() -> bool:
    for key in HF_OFFLINE_ENV_VARS:
        value = os.environ.get(key)
        if value is None:
            continue
        lowered = value.strip().lower()
        if key == "HF_DATASETS_OFFLINE_MODE":
            # "auto" can still force offline behavior depending on runtime context.
            if lowered not in {"", "0", "false", "no", "off"}:
                return True
            continue
        if _is_truthy_env(value):
            return True
    return False


def _looks_like_offline_error(exc: Exception) -> bool:
    msg = str(exc)
    lowered = msg.lower()
    return "offlinemodeisenabled" in lowered or "offline mode" in lowered


@contextmanager
def _temporarily_disable_hf_offline_env():
    saved = {key: os.environ.get(key) for key in HF_OFFLINE_ENV_VARS}
    for key in HF_OFFLINE_ENV_VARS:
        if key == "HF_DATASETS_OFFLINE_MODE":
            os.environ[key] = "0"
        else:
            os.environ[key] = "0"
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def _temporarily_disable_hf_offline_runtime():
    """
    Disable offline mode in both env vars and Hugging Face runtime constants.
    Some libraries cache offline flags at import time, so env-only changes are insufficient.
    """
    datasets_config = None
    hub_constants = None
    saved_datasets_offline = None
    saved_hub_offline = None

    try:
        from datasets import config as datasets_config  # type: ignore
        saved_datasets_offline = getattr(datasets_config, "HF_DATASETS_OFFLINE", None)
        if hasattr(datasets_config, "HF_DATASETS_OFFLINE"):
            datasets_config.HF_DATASETS_OFFLINE = False
    except Exception:
        datasets_config = None

    try:
        from huggingface_hub import constants as hub_constants  # type: ignore
        saved_hub_offline = getattr(hub_constants, "HF_HUB_OFFLINE", None)
        if hasattr(hub_constants, "HF_HUB_OFFLINE"):
            hub_constants.HF_HUB_OFFLINE = False
    except Exception:
        hub_constants = None

    with _temporarily_disable_hf_offline_env():
        try:
            yield
        finally:
            if datasets_config is not None and saved_datasets_offline is not None:
                datasets_config.HF_DATASETS_OFFLINE = saved_datasets_offline
            if hub_constants is not None and saved_hub_offline is not None:
                hub_constants.HF_HUB_OFFLINE = saved_hub_offline


def _load_dataset_with_offline_fallback(**kwargs):
    try:
        return load_dataset(**kwargs)
    except Exception as exc:
        if _looks_like_offline_error(exc):
            print(
                "[WARN] HF offline mode detected while loading dataset; "
                "retrying once with offline flags/runtime disabled. "
                f"offline_env={_has_offline_env_enabled()}"
            )
            with _temporarily_disable_hf_offline_runtime():
                retry_kwargs = dict(kwargs)
                # Ensure retry is not forced into local-files-only behavior.
                if retry_kwargs.get("download_config") is None:
                    retry_kwargs["download_config"] = DownloadConfig(local_files_only=False)
                else:
                    retry_kwargs["download_config"].local_files_only = False
                return load_dataset(**retry_kwargs)
        raise


def _iter_local_dataset_candidates(data_name: str, split: str, use_streaming: bool):
    mirror_dirs = []
    for env_key in ("ALPHA_DATASET_DIR", "HF_DATASET_MIRROR_DIR"):
        env_value = os.environ.get(env_key, "").strip()
        if env_value:
            mirror_dirs.append(Path(env_value).expanduser())

    filenames = LOCAL_MIRROR_FILENAMES.get(data_name, [])
    for mirror_dir in mirror_dirs:
        for filename in filenames:
            candidate_file = mirror_dir / filename
            if candidate_file.is_file():
                yield {
                    "path": "json",
                    "data_files": {split: str(candidate_file)},
                    "split": split,
                    "streaming": use_streaming,
                }


def _resolve_streaming_mode(data_name: str, streaming: Union[bool, str]) -> bool:
    if streaming == "auto":
        return data_name in AUTO_STREAM_DATASETS
    return bool(streaming)


def _build_load_kwargs(data_name: str, split: Optional[str], use_streaming: bool) -> tuple:
    cfg = DATASET_CONFIG[data_name]
    use_split = split or cfg["default_split"]

    kwargs = {
        "path": cfg["path"],
        "split": use_split,
        "streaming": use_streaming,
    }
    if cfg.get("name") is not None:
        kwargs["name"] = cfg["name"]
    if cfg.get("data_files") is not None:
        kwargs["data_files"] = cfg["data_files"]

    return kwargs, use_split


def _load_dataset_with_fallback(kwargs: dict, data_name: str):
    if data_name != "math_full":
        try:
            return _load_dataset_with_offline_fallback(**kwargs)
        except Exception as hub_exc:
            split = kwargs.get("split", "train")
            use_streaming = bool(kwargs.get("streaming", False))

            last_local_exc = None
            for local_kwargs in _iter_local_dataset_candidates(data_name, split, use_streaming):
                try:
                    print(
                        f"[WARN] Hub load failed for {data_name}; "
                        f"trying local mirror file: {local_kwargs['data_files'][split]}"
                    )
                    return _load_dataset_with_offline_fallback(**local_kwargs)
                except Exception as local_exc:
                    last_local_exc = local_exc

            if data_name in LOCAL_MIRROR_FILENAMES:
                expected = ", ".join(LOCAL_MIRROR_FILENAMES[data_name])
                raise RuntimeError(
                    f"Unable to load dataset '{data_name}' from Hub and local mirrors. "
                    f"Expected one of [{expected}] under ALPHA_DATASET_DIR or HF_DATASET_MIRROR_DIR. "
                    "If running with restricted network, provide a local mirror path or clear offline flags "
                    "(HF_HUB_OFFLINE/HF_DATASETS_OFFLINE/HF_DATASETS_OFFLINE_MODE)."
                ) from (last_local_exc or hub_exc)

            raise

    # Keep a few known aliases because MATH mirrors on HF occasionally move.
    math_candidates = [
        {"path": "qwedsacf/competition_math", "name": None},
        {"path": "hendrycks/competition_math", "name": None},
        {"path": "EleutherAI/hendrycks_math", "name": None},
        {"path": "lighteval/MATH", "name": "all"},
    ]

    last_error = None
    for candidate in math_candidates:
        try:
            candidate_kwargs = dict(kwargs)
            candidate_kwargs["path"] = candidate["path"]
            if candidate["name"] is None:
                candidate_kwargs.pop("name", None)
            else:
                candidate_kwargs["name"] = candidate["name"]
            return _load_dataset_with_offline_fallback(**candidate_kwargs)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        "Unable to load math_full from all known mirrors: "
        "qwedsacf/competition_math, hendrycks/competition_math, "
        "EleutherAI/hendrycks_math, lighteval/MATH"
    ) from last_error


def inspect_streaming_dataset_ALPHA(
    data_name: str,
    split: Optional[str] = None,
    streaming: Union[bool, str] = "auto",
    sample_count: int = 3,
    count_scan_limit: int = 2000,
):
    """
    Inspect dataset with a focus on streaming mode:
    - count info (metadata-first, optional bounded scan fallback)
    - preview sample prompts
    """
    if data_name not in DATASET_CONFIG:
        supported = ", ".join(sorted(DATASET_CONFIG.keys()))
        raise ValueError(
            "Unsupported dataset name: "
            f"{data_name}. Supported: {supported}"
        )

    use_streaming = _resolve_streaming_mode(data_name, streaming)
    kwargs, use_split = _build_load_kwargs(data_name, split, use_streaming)

    raw_dataset = _load_dataset_with_fallback(kwargs, data_name)

    count_source = "unknown"
    total_count = None
    try:
        if raw_dataset.info and raw_dataset.info.splits and use_split in raw_dataset.info.splits:
            total_count = raw_dataset.info.splits[use_split].num_examples
            if total_count is not None:
                count_source = "metadata"
    except Exception:
        pass

    if total_count is None and use_streaming and count_scan_limit > 0:
        scanned = sum(1 for _ in raw_dataset.take(count_scan_limit))
        if scanned < count_scan_limit:
            total_count = scanned
            count_source = "streaming_scan_exact"
        else:
            total_count = scanned
            count_source = "streaming_scan_lower_bound"

    processed_dataset = load_and_process_dataset_ALPHA(
        data_name=data_name,
        split=use_split,
        streaming=use_streaming,
    )

    preview_items = []
    if sample_count > 0:
        if use_streaming:
            preview_items = list(processed_dataset.take(sample_count))
        else:
            n = min(sample_count, len(processed_dataset))
            preview_items = [processed_dataset[i] for i in range(n)]

    return {
        "data_name": data_name,
        "split": use_split,
        "streaming": use_streaming,
        "total_count": total_count,
        "count_source": count_source,
        "samples": preview_items,
    }


def load_and_process_dataset_ALPHA(
    data_name: str,
    split: Optional[str] = None,
    streaming: Union[bool, str] = "auto",
    start_index: int = 0,
    num_instances: Optional[int] = None,
    max_samples: Optional[int] = None,
):
    """
    Load and normalize datasets into a unified schema: {'turns': list[str]}.

    Args:
        data_name: Dataset key from DATASET_CONFIG.
        split: Override dataset split. If None, use default split in config.
        streaming: True/False or "auto". "auto" enables streaming for very large datasets.
        start_index: Starting row index to slice from.
        num_instances: Number of rows to keep from start_index.
        max_samples: Optional cap on number of samples after formatting.
    """
    if data_name not in DATASET_CONFIG:
        supported = ", ".join(sorted(DATASET_CONFIG.keys()))
        raise ValueError(
            "Unsupported dataset name: "
            f"{data_name}. Supported: {supported}"
        )

    cfg = DATASET_CONFIG[data_name]
    use_streaming = _resolve_streaming_mode(data_name, streaming)
    kwargs, use_split = _build_load_kwargs(data_name, split, use_streaming)

    if start_index < 0:
        raise ValueError(f"start_index must be >= 0, got {start_index}")
    if num_instances is not None and num_instances <= 0:
        raise ValueError(f"num_instances must be > 0, got {num_instances}")

    dataset = _load_dataset_with_fallback(kwargs, data_name)
    formatter: DatasetFormatter = cfg["formatter"]

    remove_columns = list(dataset.features.keys()) if dataset.features else None

    dataset = dataset.map(
        lambda x: {"turns": formatter(x)},
        remove_columns=remove_columns,
    )

    if use_streaming:
        if start_index > 0:
            dataset = dataset.skip(start_index)
        if num_instances is not None:
            dataset = dataset.take(num_instances)
        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError(f"max_samples must be > 0, got {max_samples}")
            dataset = dataset.take(max_samples)
    else:
        # Non-streaming: handle start_index and num_instances first
        dataset_len = len(dataset)
        
        # Validate start_index
        if start_index >= dataset_len:
            # Return empty dataset instead of crashing
            dataset = dataset.select([])
        else:
            # Apply start_index and num_instances
            if start_index > 0 or num_instances is not None:
                end_index = dataset_len if num_instances is None else start_index + num_instances
                start = start_index
                end = min(end_index, dataset_len)
                if start < end:
                    dataset = dataset.select(range(start, end))
                else:
                    dataset = dataset.select([])
        
        # Then apply max_samples on the already-sliced dataset
        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError(f"max_samples must be > 0, got {max_samples}")
            dataset = dataset.select(range(min(max_samples, len(dataset))))

    if use_streaming:
        print(f"✅ Loaded {data_name} in streaming mode (split={use_split}).")
    else:
        print(f"✅ Loaded {len(dataset):,} samples from {data_name} (split={use_split}).")

    return dataset


if __name__ == "__main__":
    # Streaming inspection smoke test
    streaming_test_cases = ["math_instruct", "metamath", "magicoder"]

    for data_name in streaming_test_cases:
        print("="*60)
        print(f"📊 STREAMING INSPECT: {data_name.upper()}")
        print("="*60)
        
        try:
            report = inspect_streaming_dataset_ALPHA(
                data_name,
                streaming=True,
                sample_count=3,
                count_scan_limit=2000,
            )

            if report["total_count"] is None:
                print("🔹 Number of rows: unknown")
            elif report["count_source"] == "streaming_scan_lower_bound":
                print(f"🔹 Number of rows: >= {report['total_count']:,} (scan lower bound)")
            else:
                print(f"🔹 Number of rows: {report['total_count']:,} ({report['count_source']})")

            print("\n🔹 Sample prompts:")
            for idx, sample in enumerate(report["samples"], start=1):
                prompt_content = sample["turns"][0]
                display_prompt = (prompt_content[:220] + "...") if len(prompt_content) > 220 else prompt_content
                print(f"   [{idx}] {display_prompt}")
            
            print(f"\n✅ Kiểm tra {data_name} hoàn tất!")

        except Exception as e:
            print(f"❌ Lỗi khi xử lý bộ dữ liệu {data_name}: {e}")
        
        print("\n")