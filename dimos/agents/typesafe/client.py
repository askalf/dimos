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

"""Minimal TypeSafe System One client over the documented HTTP endpoint."""

from __future__ import annotations

import os
import time
from typing import Any

import requests

from dimos.utils.logging_config import setup_logger

logger = setup_logger()

DEFAULT_BASE_URL = "https://api.typesafe.ai"
DEFAULT_MODEL = "jev-latest"
API_KEY_ENV = "TYPESAFE_API_KEY"
_RETRY_STATUSES = {429, 500, 502, 503, 504, 529}


def choice(instructions: Any, criteria: dict[str, Any]) -> dict[str, Any]:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def noul(instructions: Any, criteria: dict[str, Any] | None = None) -> dict[str, Any]:
    q: dict[str, Any] = {"type": "noul", "instructions": instructions}
    if criteria is not None:
        q["criteria"] = criteria
    return q


class SystemOneClient:
    """`POST /v1/systemone`; returns the `answers` map keyed by question id."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout_s: float = 10.0,
        retries: int = 2,
    ) -> None:
        key = api_key or os.environ.get(API_KEY_ENV, "")
        if not key:
            raise RuntimeError(f"{API_KEY_ENV} is not set")
        self.model = model
        self.timeout_s = timeout_s
        self.retries = retries
        self._url = f"{base_url.rstrip('/')}/v1/systemone"
        self._session = requests.Session()
        self._session.headers["Authorization"] = f"Bearer {key}"
        self.last_usage: dict[str, Any] = {}
        self.last_model: str = ""

    def close(self) -> None:
        self._session.close()

    def system_one(self, state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        body = {"state": state, "model": self.model, "questions": questions}
        delay = 0.2
        for attempt in range(self.retries + 1):
            resp = self._session.post(self._url, json=body, timeout=self.timeout_s)
            if resp.status_code in _RETRY_STATUSES and attempt < self.retries:
                retry_after = resp.headers.get("retry-after")
                time.sleep(float(retry_after) if retry_after else delay)
                delay *= 2
                continue
            if resp.status_code >= 400:
                raise RuntimeError(f"TypeSafe {resp.status_code}: {resp.text[:300]}")
            data = resp.json()
            self.last_usage = data.get("usage") or {}
            self.last_model = data.get("model", "")
            answers: dict[str, Any] = data["answers"]
            return answers
        raise AssertionError("unreachable")
