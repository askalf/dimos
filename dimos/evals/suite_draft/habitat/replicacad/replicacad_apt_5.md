# ReplicaCAD — apt_5

## Executable suite

`dimos/evals/suites/habitat/replicacad/replicacad_apt_5.py`: **13 cases**, covering all template questions with `draft-reference` tags. Candidates lacking usable answers remain unscored. Distance prompts explicitly use horizontal center-to-center straight-line distance. Counts and measurements preserve this layout's differences from apt_1.

Override `REPLICACAD_DATASET_CONFIG` with the inspected config below. Default: project-local `target/habitat/data/versioned_data/replica_cad_dataset/replicaCAD.scene_dataset_config.json`; exact handle `apt_5`. Fresh seed-0 Habitat environment per case. No live eval was run; settling and sensor visibility remain review items.

## Scene reference

Root `~/Documents/habitat-sim/data/versioned_data/replica_cad_dataset/`; config `replicaCAD.scene_dataset_config.json`; original `configs/scenes/apt_5.scene_instance.json`, SHA-256 `1ca21328100b7483098b0476e999915de6c8aa3a06811f1f7e9f563f00b15ca0`. Stage `frl_apartment_stage`. Read 113 rigid placements across 83 templates plus 6 articulated instances. Same apartment family as apt_1, but use this exact scene JSON: object counts and transforms differ.

Joined rigid template names to `configs/objects/` and their visual/collision asset paths. Measured visual GLBs for sofa, TV stand, and bicycles with internal node transforms plus scene rotations/translations. Those templates have COM `[0,0,0]` and no added scale. Coordinates are meters, Y-up; distance references use X/Z transformed AABB centers.

Authored rigid placements are DYNAMIC. Confirm post-load settling or freeze policy before accepting placement-derived references. Scene JSON does not explicitly supply articulated initial joint poses, so open/closed states remain pending. Keep the four ReplicaCAD variants grouped in train/dev/held-out splitting.

## Questions from templates

Prefix `replicacad_apt_5_`. Human-validation status: **draft** for every case.

| Suffix / template | Question | Format | Reference | Evidence / score |
|---|---|---|---|---|
| bicycles / Object Counting | How many bicycles are in the scene? | Count | 2 | bike_01 and bike_02 ×1 each; `C(2)` |
| beanbags / Object Counting | How many beanbag seats are in the scene? | Count | 2 | beanbag ×2; `C(2)` |
| stools / Object Counting | How many stools are in the scene? | Count | 1 | stool_02 ×1, unlike apt_1's two; `C(1)` |
| chairs / Object Counting | How many chairs are in the scene, excluding stools and beanbags? | Count | 6 | chair_01/04/05 ×2 each; template shape validation pending; `C(6)` |
| plants / Object Counting | How many indoor potted plants are in the scene? | Count | 3 | indoor_plant_01 ×1 + indoor_plant_02 ×2; `C(3)` |
| bowls / Object Counting | How many bowls are in the scene? | Count | 4, provisional | bowl_01,02,03,06 ×1 each. apt_1 also has bowl_07, but apt_5 does not; review hidden/grouped assets; `C(4)` |
| books / Object Counting | How many individual books are in the scene? | Count | 19, provisional | book_01..06 counts 6,2,1,5,4,1. Confirm templates are individual books and all are observable; `C(19)` |
| umbrella / Object Existence | Is there an umbrella in the scene? | yes/no | yes | umbrella template ×1; `Y(yes)` |
| sofa_width / Object Dimensions | What is the approximate sofa width, in meters? | Number | 2.14 | Visual local long extent 2.140818 m at unit scale; `N(2.14,0.10,0.40)` |
| stand_height / Object Dimensions | What is the approximate TV-stand height, in meters? | Number | 0.60 | Transformed height 0.601593 m; `N(0.60,0.05,0.20)` |
| sofa_stand_distance / Object Displacement | What is the approximate straight-line distance between the sofa and TV stand, in meters? | Number | 2.61 | Horizontal center distance 2.6062; substantially different from apt_1; `N(2.61,0.20,0.8)` |
| nearest_bicycle / Object Displacement | What is the approximate straight-line distance from the sofa to the nearest bicycle, in meters? | Number | 2.77 | bike_01=2.7716 m vs bike_02=3.9171 m; `N(2.77,0.20,0.8)` |
| height_order / Spatial Ordering | Order these objects from shortest to tallest. | Ranking | CBA | A) A bicycle; B) Sofa; C) TV stand. Bikes approximately 0.972 m, sofa 0.801 m, stand 0.602 m; `R(CBA)` |

## Other potential questions

1. **[Universal Relation] Does every table have a cup on it?** yes/no; **pending** object-to-surface associations and cup occlusion; `Y(reference)`.
2. **[Object State] Are the refrigerator doors closed?** yes/no; **pending** instantiated joint state/mesh inspection, not justified by fixed_base alone; `Y(reference)`.
3. **[Passage Clearance] Can a circular robot of radius 0.25 m pass between the sofa and the nearer bicycle?** yes/no; **pending** collision geometry and a defined passage line, `Y(reference)`.
4. **[Spatial Ordering] Which is closer to the sofa by collision-free travel distance, the TV stand or the nearest bicycle?** A/B; **pending** paths/approach regions. Euclidean distances differ by only 0.165 m, so do not infer the path order or demand precision beyond the sensors.

## Evidence and scoring

Template-name prefix is `objects/frl_apartment_`. Author-side visual AABB centers: sofa `(0.917600,0.410777,5.569401)`; stand `(2.401475,0.321547,7.711962)`; bike_01 `(1.181784,0.502381,2.810442)`; bike_02 `(4.175091,0.504265,3.393949)`. Measurements use scene-graph-transformed GLBs, not object origins alone.

`E(x)=exact(x, answer.strip().upper())`; `Y(x)=exact(x, yes_no(answer))`; `C(n)=exact(n, first_number(answer))`; `N(r,t,b)=numeric(r, first_number(answer), tolerance=t, band=b)`; `R(s)=rank_order(s, ranking(answer))`. Parsing failures zero; numeric full credit ±t, linear partial to ±b; rankings pairwise. All bands tentative. Do not count `tv_object` plus `tv_screen` as two TVs. References must be checked against actual frozen/settled state, camera visibility and connected navigation space; do not reveal them to agents.
