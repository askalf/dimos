# ReplicaCAD — v3_sc1_staging_00

## Executable suite

`dimos/evals/suites/habitat/replicacad/replicacad_v3_sc1_staging_00.py`: **14 cases**, all tagged `draft-reference`: 13 template questions plus `books_exists` (no), the best estimate from the rigid inventory. Book absence is provisional because stage/articulated contents have not been exhaustively checked. Other candidates have no usable answer and remain unscored. Distance prompts specify horizontal center-to-center straight-line distance.

Override `REPLICACAD_DATASET_CONFIG` with the inspected config below. Default: project-local `target/habitat/data/versioned_data/replica_cad_dataset/replicaCAD.scene_dataset_config.json`; exact handle `v3_sc1_staging_00`. Each case gets a fresh seed-0 Habitat environment. Runtime articulated loading and visibility remain review items; no live eval was run.

## Scene reference

Root `~/Documents/habitat-sim/data/versioned_data/replica_cad_dataset/`; config `replicaCAD.scene_dataset_config.json`; scene `configs/scenes/v3_sc1_staging_00.scene_instance.json`, SHA-256 `e263378aece74b4dd5f260c4c372f5ad8eb2c5cbef9f8251c920e39c33547b67`. Stage handle `Stage_v3_sc1_staging`. Read 20 rigid-object placements across 18 templates plus 6 articulated objects (fridge, kitchen counter, cupboard, door, cabinet, chest of drawers). Rigid furniture is STATIC; articulated objects are fixed-base/DYNAMIC. Initial joint positions are not given in scene JSON.

Important count correction: this layout has **two beanbags**, not one; there is one bicycle. Matched template configs reference visual/collision GLBs. Measured selected GLBs with node transforms and scene quaternion/translation; COM `[0,0,0]` for measured objects, no extra scale. X/Z is horizontal in the Y-up meter frame. Related ReplicaCAD variants must remain grouped for dataset splitting.

## Questions from templates

Prefix `replicacad_v3_sc1_staging_00_`; metadata-derived draft references awaiting user validation.

| Suffix / template | Question | Format | Ground truth | Evidence / score |
|---|---|---|---|---|
| bicycles / Object Counting | How many bicycles are in the scene? | Count | 1 | frl_apartment_bike_01 ×1, no bike_02 placement; `C(1)` |
| beanbags / Object Counting | How many beanbag seats are in the scene? | Count | 2 | frl_apartment_beanbag ×2; `C(2)` |
| chairs / Object Counting | How many chairs are in the scene, excluding beanbags and the sofa? | Count | 2 | chair_01 ×2; `C(2)` |
| plants / Object Counting | How many indoor potted plants are in the scene? | Count | 2 | indoor_plant_01 and indoor_plant_02, each ×1; `C(2)` |
| sofa_exists / Object Existence | Is there a sofa in the scene? | yes/no | yes | sofa ×1; `Y(yes)` |
| beanbag_exists / Object Existence | Are there any beanbag seats in the scene? | yes/no | yes | beanbag placements; `Y(yes)`; alternative to count, not necessarily both in final set |
| fridge_exists / Object Existence | Is there a refrigerator in the scene? | yes/no | yes, load validation pending | Articulated `fridge` instance at `(-2.178,1.01,-1.04)`; verify URDF loads in the eval composition; `Y(yes)` |
| sofa_width / Object Dimensions | What is the approximate sofa width, in meters? | Number | 2.14 | Visual local long extent 2.140818; `N(2.14,0.1,0.4)` |
| bicycle_height / Object Dimensions | What is the approximate height of the bicycle, in meters? | Number | 0.98 | Transformed visual height 0.975325 m; `N(0.98,0.06,0.25)` |
| sofa_stand_distance / Object Displacement | What is the approximate straight-line distance between the sofa and TV stand, in meters? | Number | 4.77 | X/Z visual-center distance 4.7735 m; `N(4.77,0.25,1)` |
| sofa_bike_distance / Object Displacement | What is the approximate straight-line distance between the sofa and bicycle, in meters? | Number | 6.67 | X/Z visual-center distance 6.6747 m; `N(6.67,0.3,1.2)` |
| nearer_object / Spatial Ordering | Which object is closer to the sofa in a straight line? | Letter | A | A) TV stand; B) Bicycle. 4.7735 vs 6.6747 m; `E(A)` |
| height_order / Spatial Ordering | Order these objects from shortest to tallest. | Ranking | ACB | A) TV stand; B) Bicycle; C) Sofa. 0.6016, 0.9753, 0.8005 m; `R(ACB)` |

## Other potential questions

1. **[Object Existence] Are there any books in the scene?** yes/no; **candidate no from rigid inventory**, pending stage and articulated-mesh contents plus runtime validation. Do not substitute “zero book templates” for physical absence. `Y(reference)`.
2. **[Room Counting] How many enclosed rooms are in this layout?** Count; **pending** stage segmentation. The apartment family name and furniture count are not room metadata. `C(reference)`.
3. **[Passage Clearance] What is the largest circular robot radius that can pass through the narrowest furnished passage?** Number; **pending** full obstacle geometry and definition of relevant passages, `N(r,t,b)`.
4. **[Spatial Ordering] Which is closer to the sofa by collision-free travel distance, the TV stand or bicycle?** A/B; **pending** actual paths from a consistent approach region. Do not reuse the straight-line answer automatically.

## Evidence and scoring

Visual center positions: sofa `(3.882495,0.404238,6.099925)`; stand `(3.875946,0.327193,1.326447)`; bicycle `(4.209610,0.491883,-0.566753)`. Standard numeric references use meter units and horizontal center distance; these positions are not robot goals. Rigid origins are COM-based in this variant; inspected configs for measured objects explicitly set COM zero.

`E`: exact uppercase letter; `Y`: exact after `yes_no`; `C`: exact after `first_number`; `N(r,t,b)`: `numeric(r, first_number(answer), tolerance=t, band=b)`; `R`: `rank_order(reference, ranking(answer))`. Numeric full credit ±t, linear partial to zero at ±b, rankings correct pair fraction. Parse errors/invalid permutations zero. Human validation remains required for visibility, navigation, template semantics and instantiated articulated assets. No views or live eval were used to claim semantic counts here.
