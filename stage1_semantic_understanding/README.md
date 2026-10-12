# Stage 1: Semantic Understanding

One egocentric human video and a task prompt in; `scene_semantics.json` out: the task-relevant objects with their types and the properties the reconstruction must reproduce, and the demonstration decomposed into ordered subgoals with the acting hand, the objects involved, both hands' roles and an observable goal. Everything in it is natural language read from the video; the numbers come in later stages.

## Run

```bash
python stage1_semantic_understanding/run_stage1.py dexagent.yaml
# or, without editing the YAML:
python stage1_semantic_understanding/run_stage1.py dexagent.yaml --video my_clip.mp4 --prompt "Close the red gift box" --name giftbox
```

Needs: the model under `agent:` in `dexagent.yaml`, and a CPU. The GPU perception skills (`detect_objects`, `track_objects`, `estimate_depth`, `estimate_gravity`, `extract_hand_meshes`) are used only when their packages are installed and the frames leave a property unresolved; their interpreters are named under `environments:` in `dexagent.yaml`.

Output, under `<run.output_dir>/<run.name>/stage1/`:

```
scene_semantics.json   the deliverable
trace.jsonl           every model turn, tool call and verifier verdict
new_tools.json        skills or verifiers the agent added to the library during this run, if any
sheet_*.jpg, ...      the contact sheets and frames the agent looked at
```

## The deliverable

`schema.json` is the contract. In short:

| Key | Content |
|---|---|
| `task_prompt`, `task_summary` | the prompt, and one sentence of what happens |
| `environment` | the support surface and the camera; never objects |
| `hands.used` | which hands act |
| `objects[]` | `id`, detector-friendly `name`, `type` (rigid, articulated, deformable, other), `role`, `parts`, `appearance`, `properties` the reconstruction must show, `articulation` or `deformation` blocks, initial and final state, `evidence` with times |
| `subgoals[]` | ordered; `acting_hand`, one-motion `action`, `object_refs` (`id` or `id:part`), `motion` from a fixed vocabulary, `hand_roles` for both hands, observable `goal`, optional `observed_outcome` and `time_window_s` |
| `task_success` | the final state in words |
