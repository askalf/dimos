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

"""TypeSafe System One: one endpoint, `POST /v1/systemone`."""

from __future__ import annotations

from collections.abc import Mapping
import os
import time
from typing import Literal, TypedDict

import requests
from typing_extensions import NotRequired

API_KEY_ENV = "TYPESAFE_API_KEY"
_RETRY = frozenset({429, 500, 502, 503, 504, 529})

Text = str | Mapping[str, object] | list[object]


class ChoiceQuestion(TypedDict):
    type: Literal["choice"]
    instructions: Text
    criteria: Mapping[str, Text | None]


class NoulQuestion(TypedDict):
    type: Literal["noul"]
    instructions: Text
    criteria: NotRequired[Mapping[str, Text]]


Question = ChoiceQuestion | NoulQuestion


class ChoiceAnswer(TypedDict):
    type: Literal["choice"]
    choice: str
    confidence: float
    probabilities: dict[str, float]


class NoulAnswer(TypedDict):
    type: Literal["noul"]
    noul: float


Answers = dict[str, ChoiceAnswer | NoulAnswer]


def choice(instructions: Text, criteria: Mapping[str, Text | None]) -> ChoiceQuestion:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def noul(instructions: Text, criteria: Mapping[str, Text]) -> NoulQuestion:
    return {"type": "noul", "instructions": instructions, "criteria": criteria}


class SystemOne:
    def __init__(self, api_key: str, *, model: str = "jev-latest", timeout_s: float = 10.0) -> None:
        self._url = os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai") + "/v1/systemone"
        self._model, self._timeout_s = model, timeout_s
        self._session = requests.Session()
        self._session.headers["Authorization"] = f"Bearer {api_key}"

    def __call__(self, state: object, questions: Mapping[str, Question]) -> Answers:
        body = {"state": state, "model": self._model, "questions": questions}
        for attempt in range(3):
            r = self._session.post(self._url, json=body, timeout=self._timeout_s)
            if r.status_code in _RETRY and attempt < 2:
                time.sleep(float(r.headers.get("retry-after", 0.2 * 2**attempt)))
                continue
            if r.status_code >= 400:
                raise RuntimeError(f"TypeSafe {r.status_code}: {r.text[:300]}")
            answers: Answers = r.json()["answers"]
            return answers
        raise AssertionError("unreachable")

    def close(self) -> None:
        self._session.close()
