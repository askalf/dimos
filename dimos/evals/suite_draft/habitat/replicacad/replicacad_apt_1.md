# ReplicaCAD — apt_1

## Executable suite

`dimos/evals/suites/habitat/replicacad/replicacad_apt_1.py`: **13 cases**, covering all template questions below with `draft-reference` tags. Candidates lacking usable answers are retained but not scored. Distance prompts explicitly specify horizontal center-to-center straight-line distance, matching the documented measurement convention.

Set `REPLICACAD_DATASET_CONFIG` to the inspected config below on this machine. Default: project-local `target/habitat/data/versioned_data/replica_cad_dataset/replicaCAD.scene_dataset_config.json`; exact scene handle `apt_1`. Each case gets a fresh seed-0 Habitat environment. Physics settling, articulated asset loading, and observability remain review items; no live eval was run.

## Scene reference

Root `~/Documents/habitat-sim/data/versioned_data/replica_cad_dataset/`; configuration `replicaCAD.scene_dataset_config.json`; scene `configs/scenes/apt_1.scene_instance.json`, SHA-256 `6d718c52d7cdfc24e5c8ed1325e6633e821eeb33e29ec5fb7408f912347242ea`. Stage handle `frl_apartment_stage`. Inspected 120 rigid placements across 86 templates plus 6 articulated instances. Articulated handles: fridge, kitchen_counter, kitchenCupboard_01, chestOfDrawers_01, cabinet, door2. Joint states are not explicitly specified in this scene JSON; no open/closed answer is inferred from missing values.

Metadata references below strip the `objects/frl_apartment_` prefix from rigid template names. Individual templates in `configs/objects/` reference visual and convex collision GLBs and semantic IDs. Shapes are not semantic room polygons. Geometry measurements used trimesh scene-graph transforms and placement quaternions `[w,x,y,z]`, translations, and inspected template COM values of `[0,0,0]`; measured assets had no extra scale. Scene coordinates are meters/Y-up. Distances use horizontal transformed visual AABB centers, not navmesh paths.

**State caution:** original apt_1 objects are marked DYNAMIC; authored-placement answers must be checked against the benchmark's settled/frozen initial scene. Asset paths alone do not prove runtime placements remain identical. Keep all four selected ReplicaCAD layouts grouped in dataset splits.

## Questions from templates

ID prefix `replicacad_apt_1_`. All references are authoring drafts pending human validation.

| Suffix / template | Question | Format | Reference | Source / scoring |
|---|---|---|---|---|
| bicycles / Object Counting | How many bicycles are in the scene? | Count | 2 | `bike_01` ×1, `bike_02` ×1; `C(2)` |
| beanbags / Object Counting | How many beanbag seats are in the scene? | Count | 2 | `beanbag` ×2; `C(2)` |
| stools / Object Counting | How many stools are in the scene? | Count | 2 | `stool_02` ×2; excludes chair templates; `C(2)` |
| chairs / Object Counting | How many chairs are in the scene, excluding stools and beanbags? | Count | 6 | `chair_01`, `chair_04`, `chair_05`, each ×2; validate each template is one chair; `C(6)` |
| plants / Object Counting | How many indoor potted plants are in the scene? | Count | 3 | `indoor_plant_01` ×1, `indoor_plant_02` ×2; distinguish vase decorations; `C(3)` |
| remotes / Object Counting | How many remote controls are in the scene? | Count | 2 | `remote-control_01` ×2; small-object observability needs review; `C(2)` |
| umbrella / Object Existence | Is there an umbrella in the scene? | yes/no | yes | `umbrella` ×1; `Y(yes)` |
| books / Object Counting | How many individual books are in the scene? | Count | 21, provisional | Book template counts 01..06 = 7,2,3,4,4,1. Validate single-book geometry and hidden placements before `C(21)` |
| sofa_width / Object Dimensions | What is the approximate width of the sofa, in meters? | Number | 2.14 | Visual GLB local long horizontal extent 2.140818 m, unit scale; `N(2.14,0.10,0.40)` |
| stand_height / Object Dimensions | What is the approximate height of the TV stand, in meters? | Number | 0.60 | Transformed visual height 0.601174 m; `N(0.60,0.05,0.20)` |
| sofa_stand_distance / Object Displacement | What is the approximate straight-line distance between the sofa and TV stand, in meters? | Number | 6.04 | X/Z center distance 6.0389; `N(6.04,0.30,1.2)` |
| nearest_bicycle / Object Displacement | What is the approximate straight-line distance from the sofa to the nearest bicycle, in meters? | Number | 1.15 | Center distances bike_01=5.4941, bike_02=1.1518; minimum 1.1518; `N(1.15,0.15,0.60)` |
| height_order / Spatial Ordering | Order these objects from shortest to tallest. | Ranking | BAC | A) Sofa; B) TV stand; C) A bicycle. World heights 0.8005, 0.6012, bikes 0.9648/1.0085 m; either bicycle yields same order; `R(BAC)` |

## Other potential questions

1. **[Universal Relation] Does every chair have a cushion on it?** yes/no; **pending** support/contact associations; cushion labels alone do not tell where they rest. `Y(reference)`.
2. **[Object State] Is the refrigerator open or closed?** A) Open; B) Closed; **pending** loaded URDF joint configuration and visible door pose, `E(reference)`.
3. **[Passage Clearance] What is the largest circular robot radius that can move between the sofa and nearest bicycle?** Number; **pending** actual geometry gap, not center distance minus guessed radii; `N(r,t,b)`.
4. **[Spatial Ordering] Rank the bicycles and TV stand by collision-free travel distance from the sofa.** Ranking; **pending** distinguish bikes visually, fix a sofa approach region, and compute paths with an agreed footprint. The numerical references above are straight-line only.

## Reproducible reference evidence

Transformed center coordinates `(X,Y,Z)`: sofa `(3.893562,0.447062,1.781757)`; TV stand `(2.638832,0.321503,7.688834)`; bike_01 `(-1.596493,0.522603,1.992852)`; bike_02 `(3.587735,0.499024,2.892236)`. `objects/frl_apartment_sofa.glb`, `frl_apartment_tvstand.glb`, and `frl_apartment_bike_0{1,2}.glb` were measured, including internal node transforms. These are author-side center conventions, not navigation goals.

`E`: exact normalized letter; `Y`: exact after `yes_no`; `C`: exact after `first_number`; `N(r,t,b)`: `numeric(r, first_number(answer), tolerance=t, band=b)`; `R`: `rank_order(expected, ranking(answer))`. Numeric full credit within ±t, linear partial to zero at ±b; rankings correct pair fraction. Parse failures/invalid permutations zero. Validate actual spawn, physics state, occlusion and sensor resolution. TV screen/body templates together may be one TV, so neither is naively counted as separate televisions. Stage and URDF contents must be checked before any whole-scene negative answer.
