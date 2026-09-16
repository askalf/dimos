# ReplicaCAD — v3_sc2_staging_00

## Executable suite

`dimos/evals/suites/habitat/replicacad/replicacad_v3_sc2_staging_00.py`: **14 cases**, all tagged `draft-reference`: 12 template questions plus `beanbag_exists` (no) and `books` (0), best estimates from rigid inventory. These negative references remain provisional pending review of stage/articulated contents. Candidates without usable answers remain unscored. Distance prompts specify horizontal center-to-center straight-line distance.

Override `REPLICACAD_DATASET_CONFIG` with the inspected config below. Default: project-local `target/habitat/data/versioned_data/replica_cad_dataset/replicaCAD.scene_dataset_config.json`; exact handle `v3_sc2_staging_00`. Fresh seed-0 Habitat environment per case. No live eval was run; articulated asset loading and visibility remain review items.

## Scene reference

Root `~/Documents/habitat-sim/data/versioned_data/replica_cad_dataset/`; config `replicaCAD.scene_dataset_config.json`; scene `configs/scenes/v3_sc2_staging_00.scene_instance.json`, SHA-256 `1bd0f7de57c33dc11f266d9e8e04e168fd53b177232c57f75eaf36af0f20bf62`. Stage `Stage_v3_sc2_staging`. Inspected 19 rigid placements across 18 templates plus 6 articulated instances. Rigid objects are static; fixed-base articulated instances are dynamic with no explicit initial joint positions in the scene file.

This variant has **two bicycles and no beanbag rigid placements**. Do not copy sc1's beanbag count or distances. Template configs provide visual/collision paths and COM; measured objects have COM zero and no scale. Geometry calculations include GLB node transforms and scene quaternion/translation. All four selected ReplicaCAD variants share an apartment family and belong in the same split group.

## Questions from templates

Prefix `replicacad_v3_sc2_staging_00_`. Draft metadata/geometry references awaiting human validation.

| Suffix / template | Question | Format | Reference | Evidence / score |
|---|---|---|---|---|
| bicycles / Object Counting | How many bicycles are in the scene? | Count | 2 | bike_01 and bike_02 ×1 each; `C(2)` |
| chairs / Object Counting | How many chairs are in the scene, excluding the sofa? | Count | 2 | chair_01 ×2; `C(2)` |
| plants / Object Counting | How many indoor potted plants are in the scene? | Count | 2 | indoor_plant_01 and indoor_plant_02 ×1 each; `C(2)` |
| sofa_exists / Object Existence | Is there a sofa in the scene? | yes/no | yes | frl_apartment_sofa placement; `Y(yes)` |
| fridge_exists / Object Existence | Is there a refrigerator in the scene? | yes/no | yes, load validation pending | Articulated fridge at `(-2.178,1.01,-0.24)`; verify URDF instance appears at runtime; `Y(yes)` |
| sofa_width / Object Dimensions | What is the approximate sofa width, in meters? | Number | 2.14 | Local visual long extent 2.140818 m; `N(2.14,0.1,0.4)` |
| tvstand_height / Object Dimensions | What is the approximate height of the TV stand, in meters? | Number | 0.60 | Transformed visual height 0.601073 m; `N(0.60,0.05,0.2)` |
| sofa_stand_distance / Object Displacement | What is the approximate straight-line distance between the sofa and TV stand, in meters? | Number | 4.83 | X/Z transformed-center distance 4.8275 m; `N(4.83,0.25,1)` |
| nearest_bike / Object Displacement | What is the approximate straight-line distance from the sofa to the nearest bicycle, in meters? | Number | 5.61 | bike_02=5.6063 m, bike_01=7.1738 m; `N(5.61,0.3,1.2)` |
| farthest_bike / Object Displacement | What is the approximate straight-line distance from the sofa to the farther bicycle, in meters? | Number | 7.17 | Maximum of the two center distances; `N(7.17,0.35,1.4)` |
| nearer_object / Spatial Ordering | Which is closer to the sofa in a straight line? | Letter | B | A) Nearest bicycle; B) TV stand. 5.6063 vs 4.8275 m; `E(B)` |
| height_order / Spatial Ordering | Order these objects from shortest to tallest. | Ranking | BCA | A) A bicycle; B) TV stand; C) Sofa. Bicycle heights 0.9746/0.9753, stand 0.6011, sofa 0.8005 m; either bicycle gives same order; `R(BCA)` |

## Other potential questions

1. **[Object Existence] Is there a beanbag seat anywhere in the scene?** yes/no; **candidate no**, but validate stage/URDF contents before treating the zero rigid beanbag inventory as exhaustive. Useful contrast to sc1; `Y(reference)`.
2. **[Object Counting] How many books are in the scene?** Count; **candidate zero**, pending inspection of stage and articulated contents. Zero `book_*` placements is evidence, not proof; `C(reference)`.
3. **[Object State] Is the main door open or closed?** A/B; **pending** runtime joint state and visual interpretation of `door2`; no explicit pose in JSON, `E(reference)`.
4. **[Spatial Ordering] Which bicycle is nearer to the sofa by collision-free travel distance?** A/B; **pending** visually distinguishable bike labels and a common start/target-region path computation; numeric IDs such as bike_01 are not an agent-visible identification scheme.

## Evidence and scoring

Computed visual AABB centers, meters/Y-up: sofa `(1.572803,0.404238,4.507500)`; TV stand `(0.496757,0.322431,-0.198583)`; bike_01 `(0.668020,0.481758,-2.609061)`; bike_02 `(-0.380887,0.482159,-0.747357)`. Distances are horizontal X/Z, not collision-free paths. Rigid placement uses COM zero for these measured assets.

`E(x)=exact(x, answer.strip().upper())`; `Y(x)=exact(x, yes_no(answer))`; `C(n)=exact(n, first_number(answer))`; `N(r,t,b)=numeric(r, first_number(answer), tolerance=t, band=b)`; `R(s)=rank_order(s, ranking(answer))`. Parse errors/invalid permutations zero; numeric full-credit ±t with linear partial to zero at ±b; rankings pairwise. Tolerances provisional. Spawn, visibility and runtime configuration remain review items; ground-truth metadata stays author-side.
