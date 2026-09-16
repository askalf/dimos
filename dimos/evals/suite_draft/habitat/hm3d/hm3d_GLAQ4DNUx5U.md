# HM3D — GLAQ4DNUx5U

## Executable suite

`dimos/evals/suites/habitat/hm3d/hm3d_GLAQ4DNUx5U.py`: **14 cases**, all tagged `draft-reference`. Includes the 12 template questions plus `mural_location` (C, bedroom) and `every_desk_chair` (yes, best estimate from the two desk/chair region associations). The latter is a provisional association, not a verified support relation. Remaining candidates with no usable answer are not scored. Washer/oven category interpretation and inferred bedroom labels are retained as best-effort answers for user review.

Override `HM3D_ANNOTATED_DATASET_CONFIG`; default is the project-local annotated example configuration below, scene `00861-GLAQ4DNUx5U`. Each case gets a fresh seed-0 Habitat environment. The backend disables semantic publication; annotations are only reference evidence. No live eval or connected-island coverage check was performed.

## Scene reference

- **Full handle:** `00861-GLAQ4DNUx5U`.
- **Root:** `~/Documents/dimos/target/habitat/data/versioned_data/hm3d-0.2/hm3d/example/`.
- **Configuration:** `hm3d_annotated_example_basis.scene_dataset_config.json`, with semantic textures enabled for authoring. The agent-facing Habitat eval backend disables semantic publication.
- **Scene folder:** `00861-GLAQ4DNUx5U/` contains visual GLB, navmesh, semantic GLB, and semantic text annotations.
- **Visual GLB SHA-256:** `d2271f8fbf57aad34a2d1e03565d5bb9e85503e05eb5f8b9fd47fa66d8983577`.
- **Semantic text SHA-256:** `ef072ebe35073f3da4d319837046d76ce7aa7ccedc5aa2de2fd1e4d9c96766d3`.
- **Semantic GLB SHA-256:** `9cf5e954d7ecc7b9c41cbb66b7f4200f0965885fe940130bce3c3e1bb221a79f`.
- **Navmesh SHA-256:** `924b0d1cb33ae5d91b3fab84aa17f43f9293baccd2571b8b840441cd88a4f87f`.

`GLAQ4DNUx5U.semantic.txt` contains **907 records** of `object ID, color, category, region ID`. Habitat-Sim 0.3.3 loaded an array of 908 semantic objects (including its extra index) and 25 region slots. Text records reference region IDs 0–23. These counts are **not** room/object-instance counts without semantic filtering. Navmesh area approximately 125.911 m², five islands; neither value establishes total floor area or story count.

The metadata is the primary reference. Supplemental authoring inspection rendered 48 views from 12 seeded navigable positions with a 1.30 m camera. It shows a multi-level furnished home, utility/storage areas, bedrooms, bathrooms, and a kitchen/living level. Scan holes are visible; avoid interpreting holes as real openings.

**Measurement warning:** the loaded semantic AABBs for several objects span implausibly large ranges or extend to the origin (for example an oven box over 9 m wide). They were inspected but **not used** as furniture dimensions or object centers. Numerical geometry questions below remain pending until semantic vertex subsets or measurements are validated.

## Questions from templates

IDs use `hm3d_GLAQ4DNUx5U_<suffix>`. All references are annotation-derived drafts for human review; complete-world counts require enough agent observation coverage.

| Suffix / template | Question | Format | Draft ground truth | Source / scoring |
|---|---|---|---|---|
| beds / Object Counting | How many beds are in the scanned home? | Count | 4 | IDs 161,490,618,791 in regions 6,12,17,22; four semantic bed instances, check annotation completeness; `C(4)` |
| televisions / Object Counting | How many televisions are in the scanned home? | Count | 3 | IDs 398,480,603 in regions 10,12,17; `C(3)` |
| toilets / Object Counting | How many toilets are in the scanned home? | Count | 4 | IDs 3,175,529,636 in regions 0,7,15,18; `C(4)` |
| washing_machines / Object Counting | How many washing machines are in the utility room? | Count | 2, annotation candidate | IDs 218,219, both region 8; verify physical distinction between washer and dryer before accepting the label-based wording; `C(2)` |
| exercise_bike_exists / Object Existence | Is there an exercise bike in the home? | yes/no | yes | Semantic ID151, region6; `Y(yes)` |
| exercise_bike_location / Object Location | Which type of room contains the exercise bike? | Letter | B, provisional | A) Kitchen; B) Bedroom; C) Bathroom; D) Garage. Bike151 shares region6 with bed161; **bedroom label inferred**, verify the actual room; `E(B)` |
| fridge_location / Object Location | Which type of room contains the refrigerator? | Letter | C, provisional | A) Living room; B) Bedroom; C) Utility/laundry room; D) Bathroom. The only refrigerator-labeled instance, ID220, shares region8 with machines218/219, ironing board221 and detergent bottles; verify room interpretation visually; `E(C)` |
| ironing_board_exists / Object Existence | Is there an ironing board in the utility room? | yes/no | yes | ID221, region8 with laundry appliances; `Y(yes)` |
| vacuum_cleaners / Object Counting | How many vacuum cleaners are in the scanned home? | Count | 2 | IDs141 and227, regions5 and8; distinguish annotations from multiple views of the same unit; `C(2)` |
| bedroom_tvs / Object Counting | How many bedrooms contain a television? | Count | 2, provisional | Bed-associated regions12 and17 also contain TV480/603; bed regions6 and22 do not. Validate bedroom interpretation and missing-screen labels; `C(2)` |
| every_bedroom_tv / Universal Relation | Does every bedroom have a television? | yes/no | no, provisional | Four bed-associated regions, only two with a `tv` annotation; stronger after verifying category completeness; `Y(no)` |
| ovens / Object Counting | How many ovens are in the kitchen? | Count | 2, annotation candidate | IDs320/321 in region9; validate two actual oven units rather than mislabeled appliance faces; `C(2)` |

## Other potential questions

### 13. [Room Counting] How many distinct interior rooms are in the home?

- **Answer:** count; `C(reference)`.
- **Ground truth:** pending. Region IDs are not semantic room names; walls, stair landings, closets and other zones all contribute to the annotation regions. Do not answer 24 or 25 just from region totals.
- **Work needed:** review region boundaries against physical walls and stair structure, and adopt a consistent closet/hallway/open-plan convention.

### 14. [Object Dimensions] What is the approximate refrigerator height, in meters?

- **Answer:** number; `N(h,t,b)`.
- **Ground truth:** pending. Semantic ID220 identifies the target, but the loaded semantic AABB is not reliable enough for measurement.
- **Work needed:** isolate matching semantic geometry/texture labels, validate the vertical extent, or measure against calibrated depth and annotated instance pixels. Use transformed axes, not raw GLB axes.

### 15. [Object Displacement] How far apart are the two washing appliances, in meters?

- **Answer:** number; `N(d,t,b)`.
- **Ground truth:** pending. IDs218/219 identify the objects but their loaded bounding centers cannot be trusted.
- **Work needed:** validate geometric centers or surface separation and state the chosen metric. Region membership alone supplies no distance.

### 16. [Universal Relation] Does every desk have a desk chair?

- **Answer:** yes/no; candidate yes, **not established**. Desk559/chair556 share region16 and desk811/chair802 share region22.
- **Work needed:** confirm the chairs serve those desks and are spatially associated, rather than simply in the same room. `Y(reference)` after validation.

### 17. [Visual Attribute] Which room has a large colorful graffiti-style wall mural?

- **Options:** A) Kitchen; B) Bathroom; C) Bedroom; D) Utility room.
- **Answer:** letter; candidate **C**, `E(C)`.
- **Evidence:** supplemental view sample8 at approximately `(-8.86,1.21,0.81)`, yaw180°, shows mural, bed and desk. No mural semantic category is used for this answer.
- **Validation:** a useful distinctive visual-localization question, but requires checking actual robot-camera visibility and uniqueness of the mural.

### 18. [Passage Clearance] What is the narrowest interior doorway width, in meters?

- **Answer:** number; `N(w,t,b)`.
- **Ground truth:** pending. There are 14 `door` records, plus door frames and combined door/window records; those are not a complete architectural opening graph.
- **Work needed:** identify real jambs, ignore scan holes, measure each chosen opening and take the minimum. Do not count annotated door meshes as doorways directly.

## Evidence and validation conventions

Text annotation IDs and region membership are the durable references. Record order is not a viewing order. Dataset config specifies source up `[0,0,1]` and front `[0,1,0]`; Habitat's runtime positions are Y-up after its stage transform. The semantic data is privileged authoring data and must not enter an evaluated agent's observation stream.

Temporary inspection: `/tmp/opencode/glaq_semantics.py`, `glaq_contact0.jpg`, `glaq_contact1.jpg`; sampled seed8, 400×300 RGB, 1.30 m camera height, four headings per point. These supplemental views do not prove traversal between the five islands or visibility with the default 0.45 m camera.

`E(x)=exact(x, answer.strip().upper())`; `Y(x)=exact(x, yes_no(answer))`; `C(n)=exact(n, first_number(answer))`; `N(r,t,b)=numeric(r, first_number(answer), tolerance=t, band=b)`. Exact answers score 1/0; numeric full credit ±t with linear partial to zero at ±b; parser errors zero. No numerical reference was fabricated from suspicious semantic bounding boxes. The next review can refine overlapping questions and check complete-scene coverage.
