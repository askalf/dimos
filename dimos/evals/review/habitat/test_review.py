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

import ast
from copy import deepcopy
import math
from unittest.mock import Mock

from fastapi.testclient import TestClient
import numpy as np
import pytest

from dimos.evals.review.habitat.applied_review import current_reviews
from dimos.evals.review.habitat.catalog import build_catalog, reference
from dimos.evals.review.habitat.server import FeedbackStore, create_app
from dimos.evals.review.habitat.view_math import ceiling_cutaway, perspective


@pytest.fixture(scope="module")
def catalog():
    return build_catalog()


def test_live_suite_catalog_reference_alignment(catalog):
    assert len(catalog) == 18
    assert sum(len(s["questions"]) for s in catalog.values()) == 210
    questions = {q["id"]: q for s in catalog.values() for q in s["questions"]}
    assert questions["hssd_102344193_largest_room"]["reference"]["answer"] == "B"
    assert questions["hssd_102344193_living_area"]["reference"]["tolerance"] == 2.5
    assert questions["replicacad_apt_1_height_order"]["reference"]["answer"] == "BAC"
    assert "B) Living room" in questions["hssd_102344193_largest_room"]["options"]
    assert all(s["context"].startswith("# Habitat scenes") for s in catalog.values())
    assert all(f"## {slug}\n" in s["context"] for slug, s in catalog.items())


def test_unknown_reference_fails_instead_of_guessing():
    with pytest.raises(ValueError, match="Unsupported scorer"):
        reference(ast.parse("lambda x: custom_grader(x)", mode="eval"))


def test_applied_reviews_do_not_duplicate_questions_or_erase_history(catalog):
    scene = "hssd_108736851_177263586"
    source = "new-c35abd86-10d8-49cb-9648-5af00a6a03d8"
    target = scene + "_dining_chairs"
    retired = scene + "_bathrooms"
    history = {
        scene: {
            source: {
                "revision": 5,
                "status": "approved",
                "comment": "Eight chairs",
                "original": {"id": source, "new": True},
            },
            retired: {"revision": 1, "status": "remove", "original": {"id": retired}},
        }
    }
    before = deepcopy(history)
    visible = current_reviews(catalog, history)[scene]
    assert set(visible) == {target}
    assert visible[target]["status"] == "approved"
    assert visible[target]["revision"] == 0
    assert visible[target]["original"]["reference"]["answer"] == 8
    assert not visible[target]["original"].get("new")
    assert visible[target]["applied_comment"] == "Eight chairs"
    assert history == before
    # A later historical edit has not been applied by this validation pass.
    history[scene][source]["revision"] = 6
    history[scene][source]["comment"] = "Please check the count again"
    visible = current_reviews(catalog, history)[scene]
    assert visible[target]["status"] == "changes"
    assert visible[target]["comment"] == "Please check the count again"
    # A fresh canonical review takes precedence over the legacy placeholder.
    canonical = {"revision": 1, "status": "changes", "comment": "A new follow-up"}
    history[scene][target] = canonical
    assert current_reviews(catalog, history)[scene][target] == canonical


def test_reviews_persist_and_reject_stale_edits(tmp_path, catalog):
    path = tmp_path / "reviews.db"
    worker = Mock(current=None)
    app = create_app(catalog, FeedbackStore(path), worker)
    sid, qid = "hssd_102344193", "hssd_102344193_bedrooms"
    url = f"/api/reviews/{sid}/{qid}"
    with TestClient(app) as client:
        assert (
            client.put(
                url,
                json={"status": "changes", "comment": "Check the closet", "proposed_answer": "2"},
            ).status_code
            == 200
        )
        assert client.put(url, json={"status": "approved"}).status_code == 409
        assert (
            client.put(
                url, json={"revision": 1, "status": "approved", "comment": "Confirmed one"}
            ).status_code
            == 200
        )
        assert client.put(url, json={"revision": 2, "status": "bad"}).status_code == 422
        assert (
            client.put(
                url, json={"revision": 2}, headers={"Origin": "http://unrelated.example"}
            ).status_code
            == 403
        )
        assert client.put(f"/api/reviews/{sid}/not-a-question", json={}).status_code == 404
        exported = client.get("/api/export").json()
        assert exported["reviews"][sid][qid]["comment"] == "Confirmed one"
        assert exported["reviews"][sid][qid]["original"]["reference"]["answer"] == 1
    reopened = FeedbackStore(path).all()
    assert reopened[sid][qid]["status"] == "approved"
    assert reopened[sid][qid]["revision"] == 2


def test_new_questions_and_controls(tmp_path, catalog):
    worker = Mock(current=None)
    worker.request.return_value = {"ok": True, "state": {"mode": "topdown"}, "jpeg": "YWJj"}
    app = create_app(catalog, FeedbackStore(tmp_path / "reviews.db"), worker)
    with TestClient(app) as client:
        sid = "replicacad_apt_1"
        result = client.post(
            f"/api/questions/{sid}",
            json={
                "status": "changes",
                "proposed_question": "How many lamps?",
                "proposed_answer": "3",
            },
        ).json()
        qid = result["id"]
        assert client.get("/api/catalog").json()["reviews"][sid][qid]["original"]["new"]
        assert (
            client.put(
                f"/api/reviews/{sid}/{qid}", json={"revision": 1, "proposed_answer": "4"}
            ).status_code
            == 200
        )
        assert client.post("/api/load/nonexistent").status_code == 404
        assert client.post(f"/api/load/{sid}").status_code == 200
        with client.websocket_connect("/api/camera") as ws:
            ws.send_json({"scene": sid, "floor": 0, "cut_height": 1.8})
            assert ws.receive_json()["state"]["mode"] == "topdown"
            assert ws.receive_bytes() == b"abc"
            ws.send_json({"scene": sid, "cut_height": 100})
            assert "error" in ws.receive_json()
            ws.send_json(
                {
                    "scene": sid,
                    "mode": "orbit",
                    "orbit_yaw": 90,
                    "orbit_elevation": 45,
                    "orbit_zoom": 2,
                    "pan_x": 1.5,
                    "cutaway": False,
                }
            )
            ws.receive_json()
            ws.receive_bytes()
            assert worker.request.call_args.args[0]["mode"] == "orbit"
            assert worker.request.call_args.args[0]["cutaway"] is False
            ws.send_json({"scene": sid, "mode": "orbit", "orbit_elevation": 0})
            assert "error" in ws.receive_json()


@pytest.mark.parametrize("elevation", [10, 45, 85])
def test_orbit_cutaway_keeps_horizontal_room_slice(elevation):
    angle = math.radians(elevation)
    eye = np.array([0, 10 * math.sin(angle) + 2, 10 * math.cos(angle)])
    rotation = np.array(
        [[1, 0, 0], [0, math.cos(angle), math.sin(angle)], [0, -math.sin(angle), math.cos(angle)]]
    )
    camera_to_world = np.eye(4)
    camera_to_world[:3, :3] = rotation
    camera_to_world[:3, 3] = eye
    view = np.linalg.inv(camera_to_world)
    projection = perspective(math.radians(48), 4 / 3, 0.05, 100)
    clipped = ceiling_cutaway(projection, view, 1.8)
    # Different horizontal positions must all be cut at the same world height,
    # not a tilted camera-relative near plane.
    for x in (-1, 0, 1):
        for z in (-1, 0, 1):
            below = clipped @ view @ np.array([x, 1.0, z, 1])
            above = clipped @ view @ np.array([x, 2.5, z, 1])
            at_cut = clipped @ view @ np.array([x, 1.8, z, 1])
            assert below[2] + below[3] > 0
            assert above[2] + above[3] < 0
            assert at_cut[2] + at_cut[3] == pytest.approx(0, abs=1e-8)
