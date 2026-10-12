# Stage 3: verified robot trajectory generation

## Inputs and contract

Use the supplied human video, task prompt, Stage 1 semantics, Stage 2 scene and
robot setup. Work only in the candidate workspace specified in the task message.
Infer the strategy from these inputs and measured tool feedback. No task-specific
solution, recorded robot trajectory, or manually supplied grasp is an input.

Produce `trajectory_manifest.json`, an executed `trajectory.npz`, `phases.json`,
an explicitly named rigid execution scene, a diagnostic video, and `stage4_task/`.
These are candidates. The harness runs its independent checks at `finish` and
publishes a fresh video and immutable result bundle only when all checks pass.
A failed or incomplete candidate is never a deliverable.
Submit tool calls to act or explicitly request `finish`; a text-only reply does
not run verification. Three consecutive replies without tool actions checkpoint
the session as a backend stall, preserving the unverified candidate for recovery.

## Procedure

1. Inspect the demonstration with contact sheets and selected frames. Read the
   Stage 1 subgoals and Stage 2 body/joint map. Identify the required actors,
   objects, contacts, ordering, goal relations and final conditions. Check that
   the supplied inputs describe the same video and physical setup.
2. Define measurable success conditions for every subgoal before motion search.
   Conditions must preserve the original task. An expectation is an additional
   check, not permission to change the task or omit an inconvenient requirement.
3. Search the library for pose, grasp, contact, planning and verification tools.
   Read their schemas and source when necessary. Use human motion only as a prior;
   candidate generation and search must account for robot kinematics and geometry.
   Build hand scenes from the current object state with `hand_scene_from_recon`.
   Where supplied, use human hand observations through the library's retargeting
   and grasp-seeding tools. Otherwise use task geometry and the observed contacts.
4. Search and rank candidates, including their approach and complete intended
   manipulation path. Check arm reach, collisions, grasp stability and release,
   not just a single final wrist pose. `optimize_grasp_pose`, `arm_workspace_map`,
   `arm_reach_check` and the pose tools provide general search mechanisms.
   For a held free object, grasp-search `arm_waypoints` can describe its target
   world body positions and orientations. Include the intended reorientation
   and destination when ranking grasps; a lift-only witness is insufficient for
   a later rotation or placement. A failed local search is inconclusive, not
   proof that the task or fixed robot station is infeasible.
   Preserve the configured robot station placement and object initial states.
5. Run appropriate grasp/contact stress tests through the harness tool interface.
   Keep the exact pose, hand scene and returned `evidence` pointer with each test
   in the manifest. Testing a related pose or merely writing `ok: true` is not
   evidence. The final verifier independently reruns trusted stress tests on the
   actual inputs. Newly developed tools may help search, but cannot replace the
   trusted physical and task acceptance checks.
   For `grasp_more`, use `multi_object_hold_test` while searching candidate pairs,
   then `combined_hold_state_test` on the actual executed segment-end frame with
   every object retained by that hand. The final proof must cover the full
   combined hold in its recorded state, including the earlier grasp.
6. Plan all subgoals with the library motion vocabulary, then use
   `execute_subgoal_plan` through its advertised harness tool (a direct Python
   invocation is diagnostic only and has no trusted execution receipt). The
   complete robot executes in a rigid arm-hand model:
   fingers attach to the arm flange and the physical arm joints drive the hand.
   Floating-hand poses and keyframes are planning representations only. Use the
   executor's returned runtime scene and named runtime trajectory for all final
   audits, rendering and packaging. Never pair runtime qpos with the proposal
   scene or repair executed states by editing qpos.
   A dry-run is a kinematic screen, not an executed prefix. Deformable contacts
   require physical execution before later planning can know the object's shape.
   On `state_requires_physics`, execute the prefix and construct subsequent hand
   scenes from its recorded state. Keep grasp and arm screens on that same state.
   When search selects a pose variant, validate the returned exact `pose_used`
   and hand scene. Stress evidence for the pre-search pose does not validate a
   different executed pose. Keep the returned `execution_report` in the manifest.
7. Inspect the executor's measured outcomes. On failure, use
   `diagnose_stage3_failure` and refine the plan or candidate search. Resume from
   the earliest changed subgoal; later subgoals must execute from its resulting
   physical state. Revisit earlier choices when they prevent later goals. Batch
   independent tool calls and use search tools instead of manual angle tuning.
   Passing local contact or motion checks does not establish the task's semantic
   goal. Check the relevant goal as soon as its physical prefix exists. If the
   goal fails, revise the manipulation strategy and re-execute the affected
   prefix; do not spend the remaining budget packaging an unsuccessful motion.
   Use diagnostics of the actual object state to guide the next search. A need
   for a different plan is a reason to continue planning, not by itself a reason
   to declare the task blocked.
   The harness records diagnostic search progress and prevents a third identical
   failed execution when input contents and resume state have not changed. Use
   the measured failure to choose a different skill or repair hypothesis; changing
   an output folder does not constitute a new attempt. This guard is not an
   infeasibility verdict. Track which constraints improved or regressed, preserve
   failed attempts, and distinguish physical prefix progress from final task success.
   Read `search_progress.json` and the returned comparison: an improvement is
   reported only for comparable measured checks. Changed goals, physics, or
   plan subdivision do not count as progress. Retained `.search_prefixes/`
   artifacts are unverified proposals; inspect them to recover useful earlier
   work after a regression, then rerun the appropriate checks. If measurements
   stagnate, state a new repair hypothesis and use a relevant library skill to
   test it; repeating a failed input or changing its label is not a repair.
   Direct `optimize_grasp_pose` calls receive a search checkpoint and a cooperative
   stop marker from the harness by default. Preserve the returned checkpoint path
   and effective arguments. An administrative stop is incomplete, not a failed
   physical search or a passing pose. After resumption, remove the stop marker and
   reuse identical arguments with `resume=True`; changed inputs or framework code
   require a fresh search. Call the skill directly for this support: an arbitrary
   `run_python` wrapper is not automatically checkpointed. Cached PASS candidates
   are rechecked and cannot substitute for whole-episode verification.
   **Search instead of hand-tuning.** When a subgoal has failed twice with the
   same grasp/strategy, call `search_subgoal_variants(scene, plan, out_dir,
   prefix_dir=<that execution's out_dir>, exec_options=<the options you ran it
   with>, scene_semantic_json, scene_recon_json)`. It runs many variants of the
   failing entry's parameters as parallel physical trials from the passing prefix
   (Latin hypercube, then refits on the best), ranks them on subgoals passed,
   Stage 1 goals and the measured shortfall, and when one passes it continues on
   the next failure from that state. Give your own `knobs` (key -> [lo, hi],
   {delta_low, delta_high}, {scale}, {choices}) when you know which parameters
   matter; otherwise they are proposed from the failed checks. Then run its
   `next_step` (execute_subgoal_plan with the best plan, resume=True in the
   returned folder) so the execution is recorded, and package. A search that finds
   no passing variant means the parameters are not the problem: change the grasp,
   approach, support or subgoal split.
8. Audit the whole episode, including physical controls, collision coverage,
   continuity, task goals, synchronized actions, settling and release. Inspect
   the actual executed motion with `render_trajectory_video`; do not judge only
   selected favorable frames. For deformable objects, use the relevant shape,
   topology and integrity verifiers; visual overlap alone is not task success.
   After every `execute_subgoal_plan` the harness appends a task check of that
   execution: `stage1_goal_gate` on the recorded motion (the criterion `finish`
   applies), the best execution so far, and how many executions have passed
   without improving on it. Steer by it: when it reports no improvement, return
   to the best execution and change what its failing goal measures instead of
   re-tuning the same parameters; the search ends after a run of executions that
   do not improve. When every Stage 1 goal holds, package and finish at once.
9. Package a passing execution with `package_stage3_episode(execution_report,
   plan, ...)`: it builds the manifest from the executor's own records, runs the
   stress tests on the poses actually used, the episode audits,
   `write_stage4_config` and the video, and reports `trajectory_manifest_check`
   and `stage1_goal_gate`. (By hand: `write_stage4_config` with the execution
   scene, then the manifest with per-subgoal frame windows and evidence.) Then
   call `finish`. This runs the complete `stage3_gate_suite` and independently
   repeats required stress tests; use individual diagnostics while iterating to
   avoid duplicating the full final verification. Respond to failed checks with a framework-supported
   plan/search change. The harness supplies the same failure diagnostics to every
   task; there are no task-specific operator hints.
   Failed finish feedback includes `failed_checks` and a hashed, complete
   `diagnostic_report` for that invocation. Read the full report when feedback
   is truncated. Check both each original subgoal and final `task_success`;
   an unsupported goal clause needs an upstream contract/framework repair and
   must never be silently dropped or replaced by a weaker candidate expectation.

<!-- BEGIN rope-knot primitives -->
**Rope lying on the table (knots).** A hand holding an end cannot pass it under
a strand; tie by laying rope over rope. Motions: `lay` (pinch a cable link, lift
it just clear of the strands under its path, carry it through `via` to
`target_pos` [x, y], set it down, open with the wrist still, back out), `drag`
(the same, held 8 mm off the table, so the rope slides: a tail pulled along
itself through a crossing keeps its over/under), `pin` (the other hand holds a
link so a drag stops there) and `release`. `plan_knot_moves` predicts each move
by follow-the-leader, screens pinch, carry and landing for the arms, simulates
the cable alone and returns `lay` entries whose `expect_order` is the crossing
order the move must leave (`a+ b- c+ a- b+ c-` is an overhand); the last one
expects the `knot`. Give `moves=[{link, target, via, side}]` to plan moves you
chose (e.g. the human's). Execute one move, then plan the remaining moves again
from the executed trajectory; a `crossings` failure means the rope lies otherwise
than predicted. A `partial` result is the furthest reachable sequence, not a tie.
Prove the tie with `rope_knot_check` on the runtime scene and trajectory.
<!-- END rope-knot primitives -->

**Deformables simulated in Isaac Sim.** A Stage 2 deformable with `simulator:
"isaac"` (a rope, cable or string) is not in `scene.xml`: Isaac Sim simulates it
(its `isaac.cord_json`), MuJoCo the robot and the rigid objects. Plan and execute
the robot as usual; when the moves need the cord's links (`lay`, `drag`, `pin`,
`plan_knot_moves`), execute in the scene `build_cord_planning_scene` writes (a
MuJoCo planning copy of the cord, laid where Isaac Sim left it). Then
`execute_isaac_deformables(scene_xml=<runtime scene>, trajectory=<runtime
trajectory>, scene_recon_json, out_dir)` replays the executed robot in Isaac Sim
with the cord: a hand that closes within 3 cm of the cord holds the nearest link
with a compliant weld until it opens (grasps are read from the labels, or give
`grasps`). That recording is the cord's physics of record; what MuJoCo did to the
planning copy is never evidence. Judge each move on its mirror (`scene_mirror.xml`
+ `trajectory_mirror.npz`: `rope_knot_check`, `stage1_goal_gate`), plan the next
move from where Isaac Sim left the cord (`build_cord_planning_scene(state=
{recording, frame: -1})`), and package with `package_stage3_episode(...,
isaac_episode=<its out_dir>)`. `finish` runs the goal gates and object audits on
the mirror and `isaac_recording_check` (binding, PhysX, the cord's own physics,
an Isaac re-simulation).

## Rules

- Objects move through the simulated mechanism and contacts. Do not teleport
  objects, alter masses/friction/actuation limits, disable collisions, add grasp
  attachments absent from the real mechanism, or substitute kinematic animation
  for physical execution.
- The arm and hand remain physically connected throughout every simulation step.
  A final frame fit or a passing wrist-position check is not a substitute.
- Preserve Stage 1 object identities, targets, order and simultaneous hand roles.
  `goal_predicates` can add checks; they cannot replace an existing requirement.
  `deviation` cannot authorize different targets or sequential execution of a
  simultaneous action. Report input ambiguity upstream with visual evidence.
- Cover all manipulation subgoals. Skip only a genuinely nonmanipulation step
  already identified as such in Stage 1. A merged step must still be executed and
  checked for all of the original subgoals it covers.
- Never change verification code, trusted inputs, tolerances or reports to make
  an attempt pass. The harness freezes their identity for the run. Create a new
  reusable skill with `library_create` when the existing capabilities are
  insufficient, and test it on distinct variations as required by the library.
- If an upstream result is wrong, call `finish`, then
  `report_blocked(reason, upstream='stage1'|'stage2', evidence=...)`. A difficult
  search by itself is not evidence of an upstream error.
- A result without all mandatory measurements is unverified, not successful.
  Call `finish` or `report_blocked` within the supplied budget. Keep failures and
  attempts in the trace; do not discard failures to inflate success statistics.

## Manifest structure

Use the schema described by `trajectory_manifest_check`. Each executed entry
includes `index`, `stage1_indices`, `frames`, `motion`, `side` (or per-hand
entries), `object`, the actual `plan`, `expect`, and measured `outcome`.
For each grasp include `pose.file`, `pose.source`, `pose.hand_scene`, and
`pose.stress_tests`, retaining each test's returned `evidence: {path, sha256}`.

The manifest must identify `scene_semantic`, `scene_recon`, `scene_executed`,
`episode: {trajectory, phases, sim_video, fps, n_frames, execution_dir, execution_report}`, and
`stage4: {task_dir, config}`. `scene_executed` is the rigid runtime scene that
produced the episode. The harness checks the executed data independently of
the manifest's claimed verdicts, then renders the final video itself.
