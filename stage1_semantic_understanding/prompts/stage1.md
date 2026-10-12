# Stage 1: semantic understanding

## Contract

INPUT: one egocentric RGB video of a human performing the task, and one natural-language task prompt. Nothing else exists yet.

OUTPUT: `scene_semantics.json` in the workspace, following the schema below exactly. It states (1) every task-relevant object with its type (rigid, articulated, deformable, other) and the properties the reconstruction must reproduce, (2) the task decomposed into ordered subgoals, each with the acting hand, the objects involved, the role of both hands and an observable goal. Everything is natural language read from the video and the prompt; no measured parameters (those come in Stage 2 and 3).

Stage 2 reconstructs exactly the objects you list, with exactly the properties you name; Stage 3 optimizes one trajectory per subgoal toward the goal you write. An object you miss cannot be reconstructed, a property you omit will not exist in simulation, and a subgoal you merge or skip cannot be optimized. Accuracy of relevance, type, property and order is the whole job.

## Method

0. Use the supplied video and task description as evidence. Do not read annotations or solutions from previous runs of this sample. Infer object properties, actions and timing from the current demonstration.
1. Start from the prompt. List the objects it names or implies (the thing acted on, the tool, the target or container, the part that moves). For each, write down what type it probably is and which properties the task needs (does something slide, hinge, screw, compress, bend, get cut, leave a mark?). These are hypotheses to confirm.
2. Look at the whole clip: `probe_video`, then `make_contact_sheet` with 24 to 36 frames over the full length, then `look` at it. Note when the hands enter, what each hand touches, in which order, and the final state.
3. Zoom. For each event (a grasp, a joint moving, a release, anything small) make a contact sheet over that time window, cropped to the region, with 8 to 16 frames, and look. Small objects (cells, tips, caps) need crops. Read left and right correctly: the camera looks down the table from the operator's side, so the hand entering from the image's left is the LEFT hand.
4. Confirm or reject each hypothesis from what moves relative to what: a part that moves while the rest stays is articulation (slide, hinge, screw); a body that changes shape is deformable; a body that only translates and rotates is rigid. Record only confirmed properties; write what could not be resolved as such.
5. Use skills when the frames leave a property open: `measure_motion_timeline` for when each side of the frame (each hand) is active and for the moment of a sudden event, `detect_objects` and `track_objects` to follow an object through the clip, `estimate_depth` for relative heights and sizes, `estimate_gravity` for the table's up direction, `extract_hand_meshes` for which hand holds what. The GPU ones cost minutes; use them when they decide something.
6. Count what the task depends on (slots, drawers, latches, cups, drives, tips, the pieces of a group): zoom with a tight crop until you can count, and write the count into the properties (a group's into `pieces.count`).
7. Decompose the demonstration into subgoals in order of occurrence. One motion per subgoal (see rules). Read each subgoal's time window off the sheets or the motion timeline.
8. Write `scene_semantics.json` with `write_text`, run `scene_semantic_schema_check` and `scene_semantic_consistency_check` on it yourself, fix what they report, then call `finish`.

## Rules for objects

- Task-relevant only: the manipulated objects, the tools, the targets and containers the task interacts with, and any part that moves. No background, no clutter, no duplicates under two names. The support surface (table, cloth, mat, board the task happens on) is `environment.support_surface`, never an object, unless the task manipulates it. Hands and people are never objects.
- `id` is a short snake_case handle (`drawer_unit`, `cap_assembly`); `name` is what a detector can find in this video: colour plus noun (`grey drawer box`, `blue sports bottle`). Use the prompt's own noun when it fits (`red rope`, not `cord`).
- `type`: `rigid` moves as one piece; `articulated` has parts moving relative to each other through a joint (drawer slides, lid hinges, cap screws, blades pivot, plunger slides, spring compresses, latch flaps fold); `deformable` changes shape (rope, cloth, plush, bag, cable); `other` for liquids, granular material, things that break apart.
- `role`: `manipulated` (moved or operated), `tool` (held and used on something else), `target` (acted upon but not carried: a board written on, a mug poured into), `container` (receives something), `static` (relevant but untouched; rarely needed).
- `parts` whenever more than one part matters, always for articulated objects (the moving part and the fixed part at least). `appearance` per part when useful.
- `properties` are sentences about what must be true in simulation for the task to be possible: the mechanism and its constraints (what moves, about which axis, how far, what holds it), how it is grasped or operated (needs two hands, pressed from above, pinched), what it does on contact (leaves a mark, cuts, clamps, nests), and what stays fixed. Three to six per object; more for the main mechanism.
- Articulated objects carry `articulation` (moving_part, fixed_part, joint_type in prismatic | revolute | screw | free_part | other, axis in words, range_observed, state_over_video). Deformable objects carry `deformation` (kind, how_the_task_deforms_it).
- `evidence` cites the prompt words and the times (seconds) or frames that show the object and its type.
- A lid that stays attached while it swings is one articulated object with a `lid` part; a cap that comes off is a `screw` joint while attached and becomes its own rigid object once free (list both, as in the bottle example).
- Several separate, identical or similar pieces the task treats alike (the blocks of a tower, the cards of a pile, fan or pyramid, a group of loose bricks taken one at a time) are ONE group object with `pieces`: `count` at the first frame (`at_least: true` when only a lower bound can be counted), `kind` the singular noun of one piece, `loose: true` when they are separate bodies held together only by gravity and friction (a piece can be taken, slid or knocked off; the stack can topple), `loose: false` when they never separate in the task. `type`, `approximate_size` and the properties then describe one piece and how the pieces lie (courses turned 90 degrees, overlapping, fanned). Subgoals and `task_success` refer to the group by its id and say how many pieces ("one block piece from the loose group", "three block pieces lie side by side on the top course"); Stage 2 builds every piece as its own body and Stage 3 checks the piece the hand actually moved.
- List a piece as its own object instead when a subgoal or the final outcome names that particular piece (the queen of hearts, the red block, the top card of the pile) or it ends somewhere apart from the rest of its group; the remaining pieces stay a group with the count reduced. Do not list one object per piece when the task never tells the pieces apart, and do not describe separate pieces as one rigid body when a piece is taken from them.

## Rules for subgoals

- In order of occurrence, indexed 1..N. One motion each: grasp_lift, carry, place, release, hold, push, pull, press, open_joint, close_joint, unscrew, screw, insert, pour, tilt, cut, write, wipe, tie, fold, regrasp, other. A pick-and-place is three subgoals (grasp_lift, carry, place). A repeated motion with one checkable outcome (several snips that sever a rope, several plunger strokes, turning a cap until it is loose) is one subgoal with "repeatedly" in the action; repeated motions with separate checkable outcomes (writing 1, then 2, then 3) are separate subgoals.
- `acting_hand` is the hand that produces the motion; `both` when both act on the object at once (lifting a lid together, one hand turning while the other resists). A hand that only stabilizes is described in `hand_roles`, and a stabilizing grasp that must exist before the other hand can act gets its own `hold` subgoal first.
- `hand_roles` always describe both hands, including a resting hand ("Rests on the table.").
- Every `goal` and `task_success` clause must be measurable by Stage 3's goal gate: one relation per clause, joined
  by commas or "and", in its vocabulary ("X is held in the left hand", "X is held clear of the table", "X is above the
  Y's opening", "X rests inside Y", "X rests on Y", "X stands upright", "Y has not moved", "the lid is closed", "X is
  released", "the rope crosses over itself", "the rope has an overhand knot in it"). `goal_language_coverage_check`
  (run at finish) names any clause it cannot measure; rephrase it.
- `goal` is observable in the video and checkable in simulation: a state, not an intention ("The tray is slid out far enough that its interior is open from above", not "the drawer is opened"). When what the video shows differs from the intended outcome or cannot be resolved, write the intended outcome as the goal and add `observed_outcome`.
- `object_refs` name object ids, with `:part` for a part (`drawer_unit:sliding_tray`). `time_window_s` is read from the sheets; windows may overlap slightly and must not run backwards.
- `release` is its own subgoal only when letting go is a separate moment that matters (dropping a toy into a tray, letting a pressed lid or a seated cell go so it stays). After a `place`, the release is part of the place. Hands moving away afterwards are never a subgoal; the clip ending with empty hands needs no entry.
- `regrasp` is for a hand changing what or how it holds in a way the task needs (taking a second cup into the same hand, shifting from a carrying grip to a working grip on a tool). Settling the fingers during a pickup is part of `grasp_lift`.
- A tool returned to the table at the end is a `place` subgoal.

## Final outcome

Write `task_success` as a conjunction of observable physical states. Use separate clauses for different objects or relations; repeat the object or part name when a pronoun could refer to more than one thing. Include orientations, containment, closure, required stability, retained attachments, and whether the hands release the objects when these matter in the demonstration. An unchanged secondary object is its own requirement. A sequence of successful subgoals does not replace the final outcome.

Keep these statements qualitative. Do not write coordinates, thresholds, simulator body names, or `task_success_predicates`: numerical gate declarations are not part of Stage 1. Do not omit an observed requirement because a downstream checker cannot yet measure it; unsupported requirements must be reported for framework repair instead of silently accepted. Unknown visual or material outcomes (for example, whether a mark was erased) need their own measurements; a successful pose check does not establish them.

## Common mistakes

- Swapping left and right. Check the forearm's side of the image, not where the hand ends up.
- Calling the prompt's noun something else, or renaming an object between `objects` and `subgoals`.
- Typing an object from a single frame. Only motion across frames shows articulation.
- Treating liquid transfer, a snap, a click or a seat as observed when the frames do not show it; use `observed_outcome`.
- Merging a grasp and a carry and a place into one subgoal, or splitting one continuous motion into several.
- Listing the table, the cloth or a hand as an object.
- A tower, pile or fan of loose pieces written as one rigid object without `pieces`, or as one object per piece when the task never tells the pieces apart.

## Schema

The file must validate against this JSON Schema (`scene_semantic_schema_check` runs it):

```json
__SCHEMA__
```

## Examples

When configured below, examples illustrate the output format. Every claim about the current task still needs evidence from its supplied video.

__EXAMPLES__
