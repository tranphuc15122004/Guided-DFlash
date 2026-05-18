#!/usr/bin/env python3
"""Sequentially submit alpha data-collection jobs via the H200 bash launcher.

This script submits one job at a time by calling:
	bash script/run_h200_alpha_collect.sh

It can sweep one or more datasets and increments the start index by the
per-job instance count after each submission.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional


ROOT_DIR = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT_DIR / "script" / "run_h200_alpha_collect.sh"


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Sequentially submit alpha data-collection jobs for one or more datasets."
	)
	parser.add_argument(
		"--datasets",
		nargs="+",
		default=["metamath", "magicoder", "math_instruct"],
		help="Dataset names to run in order, e.g. gsm8k metamath math_instruct magicoder.",
	)
	parser.add_argument(
		"--offline-mode",
		type=int,
		choices=[0, 1],
		default=1,
		help="Pass OFFLINE_MODE to launcher (1=use local cache/mirror, 0=allow Hub download).",
	)
	parser.add_argument(
		"--start-index",
		type=int,
		default=0,
		help="Starting dataset index for the first job.",
	)
	parser.add_argument(
		"--num-instances",
		type=int,
		required=True,
		help="Total number of instances to collect per dataset (will be split into multiple jobs).",
	)
	parser.add_argument(
		"--instance-per-job",
		type=int,
		default=1000,
		help="Number of instances per individual PBS job.",
	)
	parser.add_argument(
		"--start-step",
		type=int,
		default=None,
		help="Override the per-job index increment. Defaults to --instance-per-job.",
	)
	parser.add_argument(
		"--walltime",
		type=str,
		default=None,
		help="Optional walltime override passed to the bash launcher.",
	)
	parser.add_argument(
		"--collector-chunk-size",
		type=int,
		default=None,
		help="Optional collector chunk size override passed to the bash launcher.",
	)
	parser.add_argument(
		"--max-new-tokens",
		type=int,
		default=None,
		help="Optional max-new-tokens override passed to the bash launcher.",
	)
	parser.add_argument(
		"--project",
		type=str,
		default=None,
		help="Optional PBS project override.",
	)
	parser.add_argument(
		"--queue",
		type=str,
		default=None,
		help="Optional PBS queue override.",
	)
	parser.add_argument(
		"--no-wait",
		action="store_true",
		help="Submit jobs without waiting for each one to finish.",
	)
	parser.add_argument(
		"--poll-seconds",
		type=int,
		default=60,
		help="Polling interval used while waiting for a submitted PBS job to finish.",
	)
	return parser.parse_args()


def _build_env(
	dataset: str,
	start_index: int,
	num_instances: int,
	args: argparse.Namespace,
) -> dict:
	env = os.environ.copy()
	env["DATASET"] = dataset
	env["START_INDEX"] = str(start_index)
	env["NUM_INSTANCES"] = str(num_instances)
	env["OFFLINE_MODE"] = str(args.offline_mode)
	if args.walltime is not None:
		env["WALLTIME"] = args.walltime
	if args.collector_chunk_size is not None:
		env["COLLECTOR_CHUNK_SIZE"] = str(args.collector_chunk_size)
	if args.max_new_tokens is not None:
		env["MAX_NEW_TOKENS"] = str(args.max_new_tokens)
	if args.project is not None:
		env["PROJECT"] = args.project
	if args.queue is not None:
		env["QUEUE"] = args.queue
	return env


def _submit_job(env: dict) -> str:
	if not LAUNCHER.exists():
		raise FileNotFoundError(f"Launcher not found: {LAUNCHER}")

	result = subprocess.run(
		["bash", str(LAUNCHER)],
		cwd=str(ROOT_DIR),
		env=env,
		text=True,
		capture_output=True,
	)

	if result.returncode != 0:
		# Print whatever output the launcher produced.
		if result.stdout:
			print(result.stdout, end="")
		if result.stderr:
			print(result.stderr, end="", file=sys.stderr)

		err_msg = (result.stderr or result.stdout or "").strip()
		raise RuntimeError(
			f"Launcher exited with code {result.returncode}.\n"
			f"{err_msg}"
		)

	if result.stdout:
		print(result.stdout, end="")
	if result.stderr:
		print(result.stderr, end="", file=sys.stderr)

	job_id = None
	for line in (result.stdout + "\n" + result.stderr).splitlines():
		match = re.search(r"Submitted job:\s*([0-9A-Za-z.\-]+)", line)
		if match:
			job_id = match.group(1)
			break

	if job_id is None:
		raise RuntimeError(
			"Could not parse PBS job ID from launcher output.\n"
			f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
		)

	return job_id


def _job_is_running(job_id: str) -> bool:
	proc = subprocess.run(
		["qstat", "-f", job_id],
		cwd=str(ROOT_DIR),
		text=True,
		capture_output=True,
	)
	if proc.returncode != 0:
		return False

	match = re.search(r"job_state\s*=\s*(\w)", proc.stdout)
	if not match:
		return False
	return match.group(1) not in {"F", "C", "E", "X"}


def _wait_for_job(job_id: str, poll_seconds: int) -> None:
	print(f"[WAIT] job {job_id}")
	while _job_is_running(job_id):
		time.sleep(poll_seconds)
	print(f"[DONE] job {job_id}")


def _ceil_div(a: int, b: int) -> int:
	return (a + b - 1) // b


def main() -> int:
	args = _parse_args()
	start_step = args.start_step or args.instance_per_job
	overall_exit = 0

	current_start = args.start_index
	for dataset in args.datasets:
		num_jobs = _ceil_div(args.num_instances, args.instance_per_job)
		remaining = args.num_instances
		job_num = 0

		print(f"[PLAN] dataset={dataset} total_instances={args.num_instances} instance_per_job={args.instance_per_job} jobs={num_jobs} start_index={current_start}")

		while remaining > 0:
			this_job_instances = min(args.instance_per_job, remaining)
			print(
				f"[SUBMIT] dataset={dataset} start_index={current_start} "
				f"num_instances={this_job_instances} (job {job_num+1}/{num_jobs})"
			)
			env = _build_env(dataset, current_start, this_job_instances, args)
			try:
				job_id = _submit_job(env)
				print(f"[SUBMITTED] {job_id}")
				time.sleep(1)

				if not args.no_wait:
					_wait_for_job(job_id, args.poll_seconds)

				current_start += this_job_instances
				remaining -= this_job_instances
				job_num += 1
			except RuntimeError as exc:
				print(f"[FAIL] {exc}", file=sys.stderr)
				overall_exit = 1
				# If a job fails to submit, skip remaining jobs for this dataset
				# (they will likely hit the same limit) and move to the next dataset.
				break

		# Reset the slice start for the next dataset unless the user is
		# intentionally sweeping a single contiguous index range.
		current_start = args.start_index

	return overall_exit


if __name__ == "__main__":
	raise SystemExit(main())
