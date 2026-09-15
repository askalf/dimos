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

"""The words `dimos data` prints around the backend's progress."""

import io

from rich.console import Console
from rich.progress import Progress

from dimos.cloud import cli


def test_squeeze_states_the_compression_only_when_it_happened() -> None:
    done = {"content_encoding": "lz4", "raw_bytes": 9_400_000, "wire_bytes": 333_900}
    assert cli._squeeze(done) == " · compressed 9.4 MB → 333.9 kB (96% smaller)"
    assert cli._squeeze({"content_encoding": None, "raw_bytes": 5, "wire_bytes": 5}) == ""
    assert cli._squeeze({}) == ""
    # already-compact data can come back no smaller; state it without a false gain
    grew = {"content_encoding": "lz4", "raw_bytes": 100, "wire_bytes": 110}
    assert cli._squeeze(grew) == " · compressed 100 bytes → 110 bytes"


def test_every_backend_phase_has_a_label() -> None:
    reported = {"compress", "checksum", "upload", "download", "verify", "decompress"}
    assert reported <= set(cli._PHASE)
    assert all(label.endswith("ing") for label in cli._PHASE.values())


def test_each_phase_is_its_own_task() -> None:
    bar = Progress(console=Console(file=io.StringIO(), force_terminal=True, width=80))
    tick = cli._Ticker(bar, "rec.db")
    assert [t.description for t in bar.tasks] == ["reading rec.db"]
    tick("compress", 50, 100)
    (t,) = bar.tasks
    assert (t.description, t.completed, t.total) == ("compressing rec.db", 50, 100)
    tick("decompress", 0, 0)
    (t,) = bar.tasks
    assert t.description == "decompressing rec.db"
    assert t.total is None, "an indeterminate phase must pulse, not show 0% of the last total"
    tick("upload", 5, 10)
    (t,) = bar.tasks
    assert (t.description, t.completed, t.total) == ("uploading rec.db", 5, 10)


def test_short_middle_elides_and_keeps_both_ends() -> None:
    name = "recording_go2_office_2026-09-08.db"
    short = cli._short(name, 30)
    assert len(short) == 30 and "…" in short
    assert short.startswith("recording") and short.endswith("09-08.db")
    assert cli._short("small.db", 30) == "small.db"  # short names pass through


def test_progress_columns_shed_speed_and_eta_when_narrow() -> None:
    from rich.progress import BarColumn, TimeRemainingColumn, TransferSpeedColumn

    wide, narrow = cli._progress_columns(120), cli._progress_columns(70)
    assert any(isinstance(c, TransferSpeedColumn) for c in wide)
    assert not any(isinstance(c, (TransferSpeedColumn, TimeRemainingColumn)) for c in narrow)
    for cols in (wide, narrow):  # the bar flexes at any width, so the line fits
        assert any(isinstance(c, BarColumn) and c.bar_width is None for c in cols)


def test_bar_is_silent_off_a_terminal(monkeypatch, capsys) -> None:
    """Piped or in CI there is no bar at all, so a log gets just the summary line."""
    from dimos.cli import theme

    monkeypatch.setattr(theme, "enabled", lambda: False)
    with cli._bar("rec.db") as tick:
        tick("upload", 1, 2)  # a no-op, must not raise
    assert capsys.readouterr().out == ""
