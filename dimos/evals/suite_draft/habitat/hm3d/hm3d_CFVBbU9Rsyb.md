# HM3D — CFVBbU9Rsyb

## Executable suite

`dimos/evals/suites/habitat/hm3d/hm3d_CFVBbU9Rsyb.py`: **13 cases**, all tagged `draft-reference` for the next user review. Includes template questions 1–11 plus best-supported candidates 19 (`sofa_color`, C) and 20 (`kitchen_cabinet_color`, A). Candidates without usable reference answers remain below and are not scored. Each case has separate context and a fresh seed-0 Habitat environment; shared landmarks do not share answer context.

Dataset override: `HM3D_DATASET_CONFIG`; default is the inspected project-local example configuration below. Full scene handle `00337-CFVBbU9Rsyb`. No live eval has been run; connected-floor coverage and default-camera visibility remain unverified.

## Scene reference

- **Habitat scene handle:** `00337-CFVBbU9Rsyb` (short ID: `CFVBbU9Rsyb`).
- **Dataset configuration inspected:** `/home/ruthwik/Documents/dimos/target/habitat/data/versioned_data/hm3d-0.2/hm3d/example/hm3d_example_basis.scene_dataset_config.json`.
- **Asset directory:** `/home/ruthwik/Documents/dimos/target/habitat/data/versioned_data/hm3d-0.2/hm3d/example/00337-CFVBbU9Rsyb/`.
- **Geometry:** `CFVBbU9Rsyb.basis.glb`; SHA-256 `ba375c4ec3112293f93be9ab2560542be068e19a900e6fed30152e33ed40ca12`.
- **Navigation:** `CFVBbU9Rsyb.basis.navmesh`; SHA-256 `76425c0f86b5e5dc2eb6b2c226168d1d820729338952e1032b9a30817e753027`.
- **GLB metadata:** glTF 2.0, generator `obj2gltf`, 374 nodes/meshes, 110 materials and 110 texture images. Names are scan fragments such as `chunk001_group000_sub028` and `window020_group000_sub025`, not an object inventory. Multiple window fragments do not imply multiple physical windows.
- **Dataset JSON:** stage configuration uses flat shading, source up `[0, 0, 1]`, front `[0, 1, 0]`, and no-lights scene defaults. No object templates are listed. The scene directory contains only the GLB and navmesh: no scene-instance JSON, XML, or semantic annotation sidecars.
- **Semantic inspection:** Habitat-Sim exposed zero semantic objects and zero semantic regions and reported failure to load a semantic descriptor. Do not assume the nearby annotated-example configuration provides annotations for this particular scene.
- **Navmesh metadata:** approximately **186.3664 m²** navigable area and **5 navigation islands**. These are navigation properties, not room count, story count, or total architectural floor area.
- **Rendered inspection:** 96 views from 24 seeded navigable positions, including sampled floor heights near Habitat Y **-2.60 m, 0 m, and +3.00 m**. There are stairs and loft-like spaces, so this is evidence for multiple elevations, not a validated total number of floors.
- **Observed character:** a multi-level furnished residential scan with repeated kitchen/living areas, bedrooms, pitched wooden ceilings, stairs, balconies, and a utility/storage room. Do not refer to “the kitchen,” “the bed,” or “the TV” without a distinguishing qualifier where multiple instances exist.
- **Status:** converted to a best-effort draft suite as described above. Evidence and remaining uncertainties are retained for the next review.

## Questions from templates

### 1. [Object Existence] Is there a washing machine in the scanned environment?

- **ID:** `hm3d_CFVBbU9Rsyb_washing_machine_exists`
- **Answer format:** `yes` or `no`.
- **Ground truth:** **Yes.**
- **Scoring:** `exact("yes", yes_no(answer))`.
- **Evidence:** V1 shows a white front-loading machine beneath a utility-room worktop.
- **Validation:** Confirm the appliance category in a close view; only presence is claimed, not a whole-building machine count.

### 2. [Object Location] Which type of room contains the washing machine beneath the long worktop?

- **ID:** `hm3d_CFVBbU9Rsyb_washing_machine_location`
- **Options:** A) Bedroom; B) Utility room; C) Living room; D) Bathroom.
- **Answer format:** One letter.
- **Ground truth:** **B — Utility room.**
- **Scoring:** `exact("B", answer.strip().upper())`.
- **Evidence:** V1–V3 show the machine with a long work surface, sink, cleaning equipment, and storage racks rather than bedroom/living furniture.
- **Validation:** Use utility/laundry/storage room as the semantic interpretation; this label is visual, not a supplied HM3D region annotation. Q1 and Q2 overlap, so choose their final inclusion deliberately.

### 3. [Object Counting] How many child high chairs stand in front of the utility-room worktop?

- **ID:** `hm3d_CFVBbU9Rsyb_utility_high_chair_count`
- **Answer format:** One count.
- **Ground truth:** **2.**
- **Scoring:** `exact(2, first_number(answer))`.
- **Evidence:** V1 shows two pale child high chairs next to one another in front of the worktop, to the left of the washing machine.
- **Validation:** Confirm both are high chairs. Scope is the pair at this worktop, not all chairs throughout the scan.

### 4. [Object Existence] Is there a wall-mounted fire extinguisher beside a stair landing?

- **ID:** `hm3d_CFVBbU9Rsyb_fire_extinguisher_exists`
- **Answer format:** `yes` or `no`.
- **Ground truth:** **Yes.**
- **Scoring:** `exact("yes", yes_no(answer))`.
- **Evidence:** V4 gives a close view of a red extinguisher mounted beside the stairs; adjacent headings from the same position show the landing.
- **Validation:** Presence is supported without needing to read its label or count all extinguishers.

### 5. [Object Counting] How many dark throw cushions are on the red sofa below the framed trousers display?

- **ID:** `hm3d_CFVBbU9Rsyb_red_sofa_cushion_count`
- **Answer format:** One count.
- **Ground truth:** **2.**
- **Scoring:** `exact(2, first_number(answer))`.
- **Evidence:** V5 shows the red sofa with one dark patterned throw cushion at each end. Built-in seat and back cushions are excluded by “throw cushions.”
- **Validation:** Confirm the distinctive framed display identifies one sofa unambiguously among the repeated living areas.

### 6. [Object Location] What is directly below the framed trousers display in the red-sofa living area?

- **ID:** `hm3d_CFVBbU9Rsyb_below_trousers_display`
- **Options:** A) Bed; B) Dining table; C) Washing machine; D) Sofa.
- **Answer format:** One letter.
- **Ground truth:** **D — Sofa.**
- **Scoring:** `exact("D", answer.strip().upper())`.
- **Evidence:** V5 shows the framed display above the red sofa.
- **Validation:** A pair-specific spatial-location question. Potentially redundant with Q5; its simpler answer format makes it a useful baseline alternative.

### 7. [Object Existence] Is there a bed beneath a skylight in a room with a sloped wooden ceiling?

- **ID:** `hm3d_CFVBbU9Rsyb_bed_below_skylight_exists`
- **Answer format:** `yes` or `no`.
- **Ground truth:** **Yes.**
- **Scoring:** `exact("yes", yes_no(answer))`.
- **Evidence:** V6 and V7 show beds below skylights in two separate upper-level areas.
- **Validation:** This is a presence question, not a claim that there is only one such bed.

### 8. [Object Existence] Are there bunk beds in the upper-level sleeping area?

- **ID:** `hm3d_CFVBbU9Rsyb_bunk_beds_exists`
- **Answer format:** `yes` or `no`.
- **Ground truth:** **Yes.**
- **Scoring:** `exact("yes", yes_no(answer))`.
- **Evidence:** V8 shows stacked sleeping platforms and a wooden bunk structure.
- **Validation:** Confirm it is a bunk bed rather than shelves. The rendered passage is narrow; check reachability from the selected eval spawn.

### 9. [Object State] Is the wardrobe in the bedroom with the blue armchair and balcony doors open or closed?

- **ID:** `hm3d_CFVBbU9Rsyb_blue_armchair_bedroom_wardrobe_state`
- **Options:** A) Open; B) Closed.
- **Answer format:** One letter.
- **Ground truth:** **B — Closed.**
- **Scoring:** `exact("B", answer.strip().upper())`.
- **Evidence:** V9 shows the wooden wardrobe front closed; V10 establishes the matching bedroom with a bed, blue armchair, and balcony doors.
- **Validation:** Confirm the object identity and both wardrobe leaves. State is baked into the scan; this does not imply an articulated wardrobe interaction is available.

### 10. [Spatial Ordering] What is the order of these locations from lowest to highest floor elevation?

- **ID:** `hm3d_CFVBbU9Rsyb_location_elevation_order`
- **Options:** A) Upper-level bunk-bed area; B) Utility room with the washing machine; C) Living area with the red sofa below the framed trousers display.
- **Answer format:** All three letters once, in order, e.g. `BCA` or `B, C, A`.
- **Ground truth:** **BCA.**
- **Scoring:** `rank_order("BCA", ranking(answer))`. Each correctly ordered pair earns one third; missing, repeated, extra, or unknown labels score zero.
- **Evidence:** Floor positions associated with V1, V5, and V8 have Habitat Y approximately **-2.60 m, 0 m, and +3.00 m**, respectively. Rank the floors supporting these locations, not the tops of furniture.
- **Validation:** This is a vertical-ordering variation of the existing spatial-ordering template. The wording “upper-level” makes one part easier; consider removing that qualifier after confirming the bunk-bed location is unique.

### 11. [Object Dimensions] Approximately how much higher is the floor of the bunk-bed area than the floor of the utility room, in meters?

- **ID:** `hm3d_CFVBbU9Rsyb_bunk_utility_height_difference`
- **Answer format:** One numerical estimate.
- **Draft ground truth:** **5.60 m.**
- **Scoring:** Proposed `numeric(5.60, first_number(answer), tolerance=0.20, band=1.0)`: full credit **5.40–5.80 m**, linear partial credit out to **4.60/6.60 m**, zero at or beyond those outer limits.
- **Evidence:** V8's navigable support position has Y=2.999462 m; V1's has Y=-2.600538 m. Difference = **5.600000 m**. This is a relative floor-elevation measurement, not ceiling height or camera elevation.
- **Validation:** Confirm these samples lie on the intended floors and the robot can observe both. Navmesh support heights are an approximate architectural reference; validate against the floor mesh before finalizing tolerance. This broadens the dimensions template to a height difference between surfaces.

## Other potential questions

### 12. [Room Counting] How many bedrooms are in the scanned environment?

- **Answer format:** One count.
- **Scoring:** `exact(reference_count, first_number(answer))` after validation.
- **Ground truth:** **Pending.** Several bedrooms, bed-containing open-plan areas, and upper-level sleeping spaces are visible. The sampled views are not a complete room inventory.
- **Reference work:** Build a floor-by-floor room map and deduplicate views of the same bedroom. Define how open sleeping areas and lofts count. Five navmesh islands do not establish five bedrooms or rooms.

### 13. [Room Size] What is the approximate area of the largest open-plan living and kitchen space, in square meters?

- **Answer format:** One numerical estimate.
- **Scoring:** `numeric(reference_area, first_number(answer), tolerance=t, band=b)`; bands pending measurement.
- **Ground truth:** **Pending.**
- **Reference work:** Segment the interior floor polygons of the repeated living/kitchen areas at different elevations. Exclude balconies and distinguish floor area from navigable area. Use geometry, not apparent size in camera views.

### 14. [Universal Relation] Does every kitchen area have a range hood above its cooking surface?

- **Answer format:** `yes` or `no`.
- **Scoring:** Exact yes/no after validating every kitchen.
- **Ground truth:** **Pending.** A hood is clear in V11; other kitchen areas have different cabinetry and need individual inspection.
- **Reference work:** Enumerate and inspect all distinct kitchens. This is a stronger universal-relation candidate than asking “every” about a single known desk, but incomplete coverage must not become a guessed answer.

### 15. [Object Dimensions] What is the approximate diagonal length of the dining tabletop beside the blue kitchen cabinets, in meters?

- **Answer format:** One numerical estimate.
- **Scoring:** Numerical tolerance-band score after geometric measurement.
- **Ground truth:** **Pending.**
- **Reference work:** V11 identifies the target. Locate its corners in the scan geometry and measure in the horizontal plane after the dataset's axis transformation. Chunk bounds do not isolate this table.

### 16. [Doorway Counting] How many interior doorways open onto the stair landing beside the fire extinguisher?

- **Answer format:** One count.
- **Scoring:** Exact count after full landing inspection.
- **Ground truth:** **Pending.**
- **Reference work:** Inspect all sides of the identified landing; define its extent and distinguish room entrances from the open stairwell. Avoid counting the same doorway from both directions.

### 17. [Passage Clearance] What is the largest circular robot radius that fits through the utility-room doorway in 2D, in meters?

- **Answer format:** One numerical estimate.
- **Scoring:** `numeric(clear_width / 2, first_number(answer), tolerance=t, band=b)`.
- **Ground truth:** **Pending.**
- **Reference work:** Identify the doorway leading into the room shown in V1–V3 and measure the clear jamb gap. Agree whether the scanned door panel affects the clearance. Do not use a navmesh corridor width without accounting for its agent-radius erosion.

### 18. [Room Connectivity] What is the minimum number of stair flights between the utility-room floor and the bunk-bed floor?

- **Answer format:** One count.
- **Scoring:** Exact count after establishing the stair graph.
- **Ground truth:** **Pending.** Vertical separation is known, but the number of flights/landings is not implied by elevation alone.
- **Reference work:** Trace the connecting stairs and define one flight consistently. Check whether these regions share a navigation island and whether the benchmark's robot can traverse the stairs.

### 19. [Visual Attribute] What color is the sofa beneath the framed trousers display in the middle-level living area?

- **Options:** A) Blue; B) Green; C) Red; D) White.
- **Answer format:** One letter.
- **Candidate ground truth:** **C — Red.**
- **Scoring:** `exact("C", answer.strip().upper())`.
- **Evidence:** V5.
- **Why separate:** Additional visual-attribute family. Use this wording rather than “What color is the red sofa?”; avoid including Q5's answer-revealing qualifier in the same case context.

### 20. [Visual Attribute] What color are the lower kitchen cabinets beside the dining table with rectangular placemats?

- **Options:** A) Blue-gray; B) Red; C) Black; D) Yellow.
- **Answer format:** One letter.
- **Candidate ground truth:** **A — Blue-gray.**
- **Scoring:** `exact("A", answer.strip().upper())`.
- **Evidence:** V11 shows blue-gray lower cabinet fronts beneath a wood-colored countertop and patterned backsplash.
- **Why separate:** Distinctive kitchen appearance can help distinguish repeated layouts. Validate color naming under the actual rendering and sensor settings.

### 21. [Object Counting] How many front-loading washing machines are in the utility room?

- **Answer format:** One count.
- **Scoring:** Exact count after checking the complete room.
- **Ground truth:** **Pending; at least one directly observed.**
- **Reference work:** V1 establishes a machine beneath the worktop but does not prove there are no other machines behind occluders or outside its view. Do not infer a total count from this positive-presence observation.

## Evidence views and reproducibility

Use the dataset configuration above when rendering: its source-axis convention matters. Positions below are **Habitat world coordinates**, Y-up, in meters. They are sampled navigable support positions, not object centers or finalized benchmark spawns. The camera was **1.30 m above each position**, **400 × 300 pixels**, with **90° horizontal FOV**. Yaw is a Habitat rotation about +Y, not a ROS yaw. Habitat-Sim version: **0.3.3**; navigable sampling seed: **6**. This inspection contains **24 positions × 4 headings = 96 views**.

| View | Sample | Position | Yaw | Evidence |
|---|---|---|---:|---|
| V1 | P14 | `(9.39473, -2.60054, -2.45872)` | 90° | Washing machine, utility worktop, two high chairs |
| V2 | P8 | `(7.02839, -2.58970, -3.33187)` | 0° | Utility sink, cleaning equipment, worktop |
| V3 | P8 | `(7.02839, -2.58970, -3.33187)` | 270° | Storage racks and tiled utility walls |
| V4 | P5 | `(2.90007, -2.60054, 3.17659)` | 180° | Fire extinguisher at stair landing |
| V5 | P23 | `(0.38860, -0.00054, -9.20557)` | 90° | Red sofa, two dark throw cushions, framed trousers display |
| V6 | P3 | `(8.30496, 2.99946, -3.10304)` | 0° | Bed under skylight and sloped wooden ceiling |
| V7 | P4 | `(-4.97951, 2.99946, -2.90696)` | 0° | Another upper-level bed/skylight area |
| V8 | P11 | `(7.78299, 2.99946, 2.87325)` | 0° | Upper-level bunk structure |
| V9 | P20 | `(4.36063, -2.60054, -7.89973)` | 180° | Closed wooden wardrobe |
| V10 | P20 | `(4.36063, -2.60054, -7.89973)` | 0° | Matching bedroom, blue armchair, balcony doors |
| V11 | P2 | `(-3.38262, -0.00054, -2.12816)` | 90° | Blue-gray kitchen cabinets, hood, dining table and placemats |

Local temporary artifacts: `/tmp/opencode/hm3d_cf_views.py`, `/tmp/opencode/cf_contact0.jpg`, `cf_contact1.jpg`, `cf_contact2.jpg`, and individual `/tmp/opencode/cf_p<sample>_<heading-index>.jpg` files (heading indices 0–3 correspond to 0°, 90°, 180°, 270°). These files are not committed assets; source hashes and view parameters above support re-rendering.

## Scoring and validation notes

- Reuse `exact`, `first_number`, `yes_no`, `numeric`, `ranking`, and `rank_order` from `dimos/evals/scorers.py`; catch parser `ValueError` and score unparseable answers zero. Keep ground truths and evidence metadata author-side.
- Yes/no, counts, and single-choice questions receive 1 for a correct answer and 0 otherwise. Numerical measurements use a full-credit tolerance with linear partial credit to the outer band. Rankings use correctly ordered pair fractions.
- Several questions deliberately share landmarks. Before finalizing, choose a nonredundant subset and keep per-case context separate so one question's wording does not reveal another's answer.
- Inspecting sampled points on five navmesh islands does **not** establish that a single robot can reach every sampled point from one fixed spawn. Confirm connected-region coverage and stair traversal before whole-building tasks. Otherwise scope questions to a selected reachable region or explicitly change the episode setup.
- Authoring camera height was 1.30 m; the current default DimOS Habitat robot camera is 0.45 m. Skylights, high cabinets, framed displays, and tabletop objects need visibility checks with the actual robot sensor. Ground-truth existence and agent observability are different checks.
- Multi-level scans need per-floor geometry and correct dataset-frame transforms for distances and areas. Do not compute whole-building area from a single top-down bounding rectangle.
- The observed furniture is baked into scan geometry. Do not assume refrigerator/wardrobe articulation exists merely because an object looks open or closed. Counterfactual-state questions require additional geometry or a separate justified reference.
- Exact whole-house absence, total room counts, doorway widths, and furniture dimensions are intentionally not guessed from generic chunk names. Those stay pending until grounded evidence exists.
