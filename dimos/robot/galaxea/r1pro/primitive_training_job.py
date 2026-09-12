# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Detached training stages with durable progress and checkpoint-friendly resume markers."""

import os
from pathlib import Path
import shutil
import subprocess
import time

from dimos.robot.galaxea.r1pro.demo_collect_objects import save_manifest


class TrainingStages:
    def __init__(self, root: Path, job: Path) -> None:
        self.root = root
        self.job = job
        self.env = {
            **os.environ,
            "PYTHONPATH": str(self.root),
            "MUJOCO_GL": "egl",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "HF_HUB_OFFLINE": "1",
            "WANDB_MODE": "disabled",
        }
        self.env.pop("DISPLAY", None)
        # One isolated learner process at a time avoids changing its environment
        # underneath another conversion/train/evaluation process.
        self.learned = [
            "uv",
            "run",
            "--offline",
            "--frozen",
            "--project",
            str(self.root / "dimos/imitation/policy/lerobot/python"),
            "--with-editable",
            str(self.root),
            "--with",
            "mujoco==3.10.0",
            "--with",
            "roboplan==0.6.0",
            "python",
        ]

    def __call__(self, name: str, command: list[str]) -> None:
        if (self.job / f"{name}.done").exists():
            return
        if shutil.disk_usage(self.job).free < 12 * 1024**3:
            raise RuntimeError(
                "Less than 12 GiB free; preserving all existing data and stopping this job"
            )
        began = time.time()
        status = dict(
            stage=name,
            state="running",
            pid=os.getpid(),
            started=began,
            log=str(self.job / f"{name}.log"),
        )
        save_manifest(self.job / "status.json", status)
        with (self.job / f"{name}.log").open("a") as log:
            process = subprocess.Popen(
                command,
                cwd=self.root,
                env=self.env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            save_manifest(self.job / "status.json", {**status, "child_pid": process.pid})
            last = began
            while process.poll() is None:
                time.sleep(30)
                if time.time() - last >= 300:
                    last = time.time()
                    save_manifest(
                        self.job / "status.json",
                        {
                            **status,
                            "child_pid": process.pid,
                            "elapsed_seconds": round(last - began),
                        },
                    )
                    print(f"{name}: {round(last - began)} seconds", flush=True)
            if process.returncode:
                raise RuntimeError(
                    f"Stage {name} exited {process.returncode}; inspect {self.job / (name + '.log')}"
                )
        (self.job / f"{name}.done").write_text("completed\n")
        print(f"Completed {name}", flush=True)
