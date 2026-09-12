# Query plan

Where the query is going, and why. Written 2026-09-12 with Jeff, working through the
current design step by step. Nothing here is built yet unless it says so.

**Target: a 100 ms query.** Today "soda" over 3,462 embedding frames takes 92.5 s.

## What the query does now

1. Turn the text into a vector, one per model.
2. Load every embedding frame's patch grids from the db.
3. Score every patch; keep the ones that beat their background.
4. For each hot patch, look up where the camera was and how far away that patch was,
   and draw a thin shell at that distance along its ray.
5. Add the shells into a voxel grid.
6. Clean up: drop voxels with no depth behind them, drop lonely ones, carve to object
   size, group what is left into blobs.
7. Return the blobs, ranked.

## Measured, on grocery.db, 3,462 embedding frames

| | |
|---|---|
| load the embedding frames | 1.2 s |
| answer without refine | 9.9 s |
| answer with refine | 92.5 s |
| build the tf buffer | 4.1 s, once per process |
| one tf lookup | 141 us, so 0.85 s for 6,000 hot patches |
| resident memory | 6.3 GB |

Two things dominate, and neither is the search:

**Refine is 89% of the query, and its cost is the wrong shape.** `refine.Grid` allocates
a dense box over the hot voxels' bounding box. For "soda" that box is 557x385x151 =
32.4 million cells holding 7,807 hot voxels: 0.024% occupancy. It is
O(bounding-box volume), not O(hot voxels), so spreading the same heat across a larger
store grows it cubically while the useful data does not move.

**Everything is loaded because of how it is stored.** The numbers the query needs live
inside one pickled blob per embedding frame, so reading one patch means unpickling the
whole row -- 6.3 GB of them.

## The layout

Three kinds of thing, three streams.

**depth_thumbnails** -- one row per embedding frame: a small point cloud in the camera's own
frame, plus the camera frame and timestamp needed to place it. Nothing else: once each
patch carries its own ray and depth, the occupancy check is the only thing that needs a
per-frame row at all, so the stream is named for the one job it does.

**patches, one stream per model** -- `hyperspace_patches__m_<model>`, one row per patch:
the vec0 vector, plus camera frame, timestamp, cell number, ray direction and depth.
Everything needed to place a hit, so a search result needs no second read. The per-model
streams are shipped (commits 7fa21a969, 25f617f46); the self-contained row is not.

What leaves today's embedding-frame row, and why: `grid`/`grids` become the per-model patch
streams; `intrinsics` is unnecessary once every patch and every thumbnail point carries
its own direction; `rows`/`cols`/`grid_shapes` collapse into the one fixed cell grid;
`thumbnail_mm`/`thumbnail_stride` become the point cloud; and
`members`/`member_specs`/`model` go because the stream name already says which model
wrote the row -- which also disposes of the misnamed `model` field rather than renaming
it.

**tf** -- the recording's own stream, untouched.

### A fixed cell grid, chosen independently of the models

Say 48x48 for every embedding frame, with each model's patches resampled onto it at ingest. Then
cell 231 means the same place in every model, so any subset of models can be combined at
runtime and a fourth model can be added next month without touching the first three.

Today the shared grid is derived from whichever models are in the ensemble, which is
what forces them to be ingested as a group.

### The thumbnail is a point cloud, not a depth picture

The occupancy step is the only thing that reads it: every embedding frame's depth is placed
into the world to give a rough "we saw a surface here" cloud, and answer voxels that are
not on it get dropped. That is what removes hits floating in mid-air. It deliberately
uses the whole frame rather than the hot patches, so it is a check rather than a
restatement of what the patch already claimed.

Storing it as 3D points in the CAMERA's frame instead of a depth raster means nothing
has to be unprojected at query time: no intrinsics, no ray table. It costs three numbers
per point instead of one, about 3x as int16 millimetres.

The camera's world pose must NOT be baked in. That would remove the tf lookup too, and a
loop closure would then silently leave every thumbnail wrong -- the same reason the tf
design below stores ids rather than values.

## The query

1. Encode the text once per model.
2. Ask each model's vec stream for its nearest patches.
3. Keep the patches that came back from every model -- which is very nearly what
   min-pooling means, a cell all the models like.
4. Walk that list of ids. Each one carries its own frame, timestamp, depth and ray, so
   it becomes a point in the world without reading anything else.
5. Add its shell into the voxels.
6. Clean up and group.

Steps 2-5 stream: take an id, place it, move on. Step 6 cannot -- finding connected
components needs the finished map, and the score normalisation needs to know the top
score. So: stream 2-5, batch 6.

### sqlite-vec is brute force

There is no index. `vec0` compares the query against every stored vector on every
search: O(N*d) plus a top-k heap. Asking for the top 10 costs the same as the top 10,000.

Two things follow. Growing k to find the threshold is free, but it also buys nothing.
And the win from searching in the db is not that the search is clever -- it is that
reading 2M vectors sequentially inside sqlite is far cheaper than unpickling 3,462 blobs
into our own memory.

## tf

Building the buffer costs 4.1 s and holding it is the load-everything approach that
Jeff's June work (`cfe1d7115`, long task BottomReptile) replaced -- and it cannot stay,
because tf has to handle live loop closures.

That design: each tf message stores a tree snapshot whose **edges are keys into the tf
table, not copied values**. A lookup walks the tree and fetches the few rows it
references -- measured then at 4 rows per query instead of loading all 38,370, 13.3x
faster. Because the edges are keys, a correction that EDITS a row propagates to every
snapshot referencing it, so there is nothing to invalidate. That is also why a
`DeformationNode` must edit existing messages rather than append corrective ones.

For hyperspace: we already walk the whole tf stream at ingest, so build it there. Per
embedding frame, store the row ids of the two tf entries bracketing its timestamp for each edge of
the camera->odom chain, about ten ids. At query, fetch those rows, interpolate, compose.
No buffer.

Ids not poses is the whole trick.

Two caveats. It only covers the chains precomputed at ingest, not an arbitrary
`tf.get(target, source, ts)`. And **mem2's `Stream` has no fetch-by-id** -- it has
`at(t)`, `tags()`, `order_by`, seek and time selectors, but nothing to fetch a set of
row ids. Jeff's June work added `fetch_by_ids` for exactly this, and it is not in dimos
today. Resolving by `at(ts)` instead would work but resolves by time rather than by row,
so corrections stop propagating, which defeats the point.

Cheap win available before any of that: patches from one embedding frame share a
timestamp, so look up once per frame rather than once per patch. Better still, every
timestamp is known up front, so sort them and interpolate the whole batch in numpy in
one pass -- batching beats threading here, where each lookup is a few tiny matrix ops
and the cost is Python call overhead the GIL holds anyway.

Note that `placer()` calls `self.tf.get` once per frame against a `MultiTBuffer` today,
and core has no batched `get`. Both the batch and the tree mean doing the interpolation
in hyperspace instead, so `self.tf.get` goes away either way.

## What each piece needs

| | db | core |
|---|---|---|
| per-model vector streams | yes (shipped) | no |
| batched tf interpolation | no | no |
| patch carries frame, ts, depth, ray | yes, re-ingest | no |
| photos stream, point-cloud thumbnail | yes, re-ingest | no |
| fixed model-independent cell grid | yes, re-ingest | no |
| tf tree of row ids | yes | **yes** -- `fetch_by_ids` |
| sparse refine | no | no |

## Still open

- Refine on the sparse set instead of a dense box, or capped the way memory_world caps
  it (`REFINE_MAX_VOXELS`, `REFINE_MAX_EXTENT_M`).
- `hot_patches` is pinned at its 6,000 cap against 1.56M patches searched.
- Recall: with three models, the 2nd-lowest beats the lowest -- P 0.73 R 0.88 against
  min's P 0.74 R 0.71. Untested: whether min is simply degenerating to whichever model
  scores lowest overall, which per-model z-scoring would fix.
- The tf that an in-place ingest used to copy into the recording (fixed 2026-09-12: it
  had duplicated grocery.db's tf 5.3x). The dedupe is separate from the layout work.
