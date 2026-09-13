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

"""The patch index held in memory, because sqlite is what a query spends its time on.

Measured on sf_office's so400m index, 886,032 patches: the vector search took 3.2 s and
reading the winning vectors back another 2.6 s, while the arithmetic those five seconds
exist to perform is 58 ms. The index is not big -- it is only *far away*. Held in an
array it answers exactly, with no top-k cap and no approximation, in the time the
maths takes.

The cost is moved rather than removed: reading a model's vectors out of vec0 is minutes,
because a virtual table hands back one row at a time. That is paid once, at startup or
as a recording is ingested, which is the trade this module exists to make.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from dimos.utils.logging_config import setup_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = setup_logger()

# Rows pulled from the vector table per round trip. Only bounds peak memory during the
# load; the table is read start to finish either way.
READ_CHUNK = 50_000

# Rows promoted to single precision at a time when scoring. Big enough that the matrix
# multiply is worth starting, small enough that the promotion is half a gigabyte.
SCORE_CHUNK = 100_000

# What the vectors are held as. Half precision because the index is what has to fit:
# grocery's three models are 34 GB at single precision and 17 at half, and the scores
# it produces are a similarity used to rank and threshold, not a measurement -- the
# matmul accumulates in float32 either way.
HELD_AS = np.float16


@dataclass
class ResidentPatches:
    """One model's patches: the vectors, and the little that placing them needs.

    Everything is parallel by row, so a patch is an index into all of them at once and
    scoring is one matrix multiply over the whole index.
    """

    tag: str
    stream: str
    vectors: NDArray[Any]
    camera_frames: list[str]
    frame_of: NDArray[np.int32]
    ts: NDArray[np.float64]
    cell: NDArray[np.int32]
    grid: NDArray[np.int16]
    ray: NDArray[np.float32]
    depth: NDArray[np.float32]

    @property
    def rows(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def width(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def megabytes(self) -> float:
        return float(self.vectors.nbytes) / 1e6

    def scores(
        self, query: NDArray[np.float32], background: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Every patch's contrast against the query: how much better than generic room.

        Exact, over every patch, rather than over whatever an approximate index would
        have returned.

        Done a block at a time, cast up to single precision as it goes. Numpy has no
        fast half-precision matrix multiply -- it falls back to something thirty times
        slower than the single-precision one, measured, which turned a 0.2 s search
        into 18 s -- while a block of a hundred thousand rows costs half a gigabyte to
        promote and runs at full speed. The storage stays halved either way.
        """
        # The query and the backgrounds go through together: one pass over the index
        # rather than one for the words and another for the room.
        texts = np.vstack([np.asarray(query, dtype=np.float32)[None, :], background])
        out = np.empty(self.rows, dtype=np.float32)
        for start in range(0, self.rows, SCORE_CHUNK):
            block = np.asarray(self.vectors[start : start + SCORE_CHUNK], dtype=np.float32)
            against = block @ texts.T
            out[start : start + len(block)] = against[:, 0] - against[:, 1:].max(axis=1)
        return out

    def hot(
        self,
        query: NDArray[np.float32],
        background: NDArray[np.float32],
        *,
        threshold: float,
        limit: int | None = None,
    ) -> tuple[NDArray[np.intp], NDArray[np.float32]]:
        """Rows scoring above *threshold*, strongest first, at most *limit* of them."""
        found = self.scores(query, background)
        picked = np.flatnonzero(found > threshold)
        order = np.argsort(-found[picked])
        if limit is not None and len(order) > limit:
            order = order[:limit]
        picked = picked[order]
        return picked, found[picked]


def _warn_if_it_will_not_fit(tag: str, rows: int, width: int) -> None:
    """Say so before spending minutes on a read that ends in a swap storm.

    Not a refusal: how much of a machine an index may have is the caller's business,
    and a model that only just fits is a normal thing to want. But finding out by
    watching the machine die is not, so the number goes in the log first.
    """
    wanted = rows * width * HELD_AS().itemsize
    try:
        import psutil

        free = int(psutil.virtual_memory().available)
    except Exception:
        # Whatever went wrong asking, it is not a reason to refuse to load an index.
        return
    if wanted > free * 0.8:
        logger.warning(
            f"hyperspace: {tag} wants {wanted / 1e9:.1f} GB resident and this machine has "
            f"{free / 1e9:.1f} GB free -- expect swapping, or pass --no-resident"
        )


def _from_chunks(conn: Any, stream: str, width: int, rows: int) -> NDArray[Any] | None:
    """Read the vectors out of vec0's chunk storage, or None if that is not possible.

    Fifty times faster than the virtual table -- 0.7 s against 34 s for half a gigabyte
    -- because a chunk is thousands of vectors in one blob where the table hands back
    one row per round trip.

    It is sqlite-vec's own storage and not an interface, so nothing here trusts it:
    the tables have to exist, and the rowid-to-slot map has to say the rows sit in
    order, or this returns None and the slow read stands. A version of sqlite-vec that
    lays them out differently makes this refuse rather than lie.
    """
    chunks = f"{stream}_vec_vector_chunks00"
    for table in (chunks, f"{stream}_vec_chunks", f"{stream}_vec_rowids"):
        found = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
        ).fetchone()
        if not found:
            return None

    sizes = dict(conn.execute(f'SELECT chunk_id, size FROM "{stream}_vec_chunks"'))
    start_of: dict[int, int] = {}
    at = 0
    for chunk_id in sorted(sizes):
        start_of[chunk_id] = at
        at += int(sizes[chunk_id])
    slots = [
        start_of[chunk_id] + offset
        for chunk_id, offset in conn.execute(
            f'SELECT chunk_id, chunk_offset FROM "{stream}_vec_rowids" ORDER BY rowid'
        )
    ]
    if len(slots) != rows or slots != list(range(rows)):
        return None

    out = np.empty((rows, width), dtype=HELD_AS)
    at = 0
    # One chunk at a time, cast as it lands: the whole table in single precision is
    # sixteen gigabytes for a big model, and the point of this is to hold half that.
    for (blob,) in conn.execute(f'SELECT vectors FROM "{chunks}" ORDER BY rowid'):
        block = np.frombuffer(blob, dtype=np.float32).reshape(-1, width)
        taken = min(len(block), rows - at)
        if taken <= 0:
            break
        out[at : at + taken] = block[:taken]
        at += taken
    return out[:at] if at == rows else None


def _vectors_of(conn: Any, stream: str, width: int, rows: int) -> NDArray[Any]:
    """Read a whole vec0 table into one array, in rowid order.

    One row per round trip through the virtual table, which is minutes for a big
    model. Only reached when the chunk storage could not be read.
    """
    out = np.empty((rows, width), dtype=HELD_AS)
    cursor = conn.execute(f'SELECT embedding FROM "{stream}_vec" ORDER BY rowid')
    at = 0
    while at < rows:
        block = cursor.fetchmany(READ_CHUNK)
        if not block:
            break
        taken = min(len(block), rows - at)
        out[at : at + taken] = np.frombuffer(
            b"".join(row[0] for row in block[:taken]), dtype=np.float32
        ).reshape(taken, width)
        at += taken
    return out[:at]


def load(store: Any, tag: str, stream: str) -> ResidentPatches:
    """Pull one model's whole patch index into memory.

    Minutes for a big model, and that is the point of doing it once. Vectors come from
    the vector table; the rest comes from the same rows' payloads, which are cheap
    (about six microseconds each) next to the vectors.
    """
    # Reaching past the Stream for the backend and the connection. There is no public
    # way to read a whole vector table or to fetch a payload by id -- `Stream.filter` is
    # a python predicate over everything, which is worse than what this replaces -- and
    # `stored_vectors` in frames.py already does the same. Worth a public accessor if
    # anything else comes to want one.
    backend = store.stream(stream, dict)._source
    blobs, codec = backend.blob_store, backend.codec
    conn = store._registry_conn

    rows = int(conn.execute(f'SELECT COUNT(*) FROM "{stream}"').fetchone()[0])
    if not rows:
        raise ValueError(f"{stream!r} holds no patches")
    ids = [row[0] for row in conn.execute(f'SELECT id FROM "{stream}" ORDER BY id')]

    started = time.monotonic()
    first = codec.decode(blobs.get(stream, ids[0]))
    probe = conn.execute(f'SELECT embedding FROM "{stream}_vec" LIMIT 1').fetchone()
    width = len(np.frombuffer(probe[0], dtype=np.float32))

    _warn_if_it_will_not_fit(tag, rows, width)
    vectors = _from_chunks(conn, stream, width, rows)
    if vectors is None:
        logger.warning(
            f"hyperspace: {tag} has no readable vec0 chunk storage, falling back to the "
            "row-by-row read -- minutes rather than seconds"
        )
        vectors = _vectors_of(conn, stream, width, rows)
    read = time.monotonic() - started

    started = time.monotonic()
    names: dict[str, int] = {}
    frame_of = np.empty(rows, dtype=np.int32)
    stamps = np.empty(rows, dtype=np.float64)
    cells = np.empty(rows, dtype=np.int32)
    grids = np.empty((rows, 2), dtype=np.int16)
    rays = np.empty((rows, 2), dtype=np.float32)
    depths = np.empty(rows, dtype=np.float32)
    for at, row_id in enumerate(ids):
        payload = first if at == 0 else codec.decode(blobs.get(stream, row_id))
        name = str(payload["camera_frame"])
        if name not in names:
            names[name] = len(names)
        frame_of[at] = names[name]
        stamps[at] = float(payload["ts"])
        cells[at] = int(payload["cell"])
        grids[at] = payload["grid"]
        rays[at] = payload["ray"]
        depths[at] = float(payload["depth"])
    meta = time.monotonic() - started

    logger.info(
        f"hyperspace: {tag} resident -- {len(vectors)} x {width} "
        f"({vectors.nbytes / 1e6:.0f} MB) in {read:.1f}s, payloads in {meta:.1f}s"
    )
    return ResidentPatches(
        tag=tag,
        stream=stream,
        vectors=vectors,
        camera_frames=[name for name, _ in sorted(names.items(), key=lambda kv: kv[1])],
        frame_of=frame_of,
        ts=stamps,
        cell=cells,
        grid=grids,
        ray=rays,
        depth=depths,
    )


class ResidentIndex:
    """The models a process is holding, loaded once and asked many times.

    Keyed by stream rather than by store so that two handles on one recording share the
    array instead of each paying the read.
    """

    def __init__(self) -> None:
        self._held: dict[str, ResidentPatches] = {}

    def __contains__(self, stream: str) -> bool:
        return stream in self._held

    def get(self, stream: str) -> ResidentPatches | None:
        return self._held.get(stream)

    def of(self, store: Any, tag: str, stream: str) -> ResidentPatches:
        held = self._held.get(stream)
        if held is None:
            held = self._held[stream] = load(store, tag, stream)
        return held

    def warm(self, store: Any, members: Sequence[tuple[str, str]]) -> float:
        """Load every named model. Returns the seconds spent, for a caller to report."""
        started = time.monotonic()
        for tag, stream in members:
            self.of(store, tag, stream)
        return time.monotonic() - started

    def drop(self, stream: str) -> None:
        self._held.pop(stream, None)


# One per process. A query path looks here before it reaches for sqlite.
RESIDENT = ResidentIndex()
