# Stage 3: Robot Trajectory Generation

Stage 1's subgoals and Stage 2's scene in; one robot episode out, verified in contact physics and packaged for Stage 4.

## Run

```bash
python stage3_trajectory_generation/run_stage3.py dexagent.yaml                 # after Stage 1 and Stage 2 of the same run
python stage3_trajectory_generation/run_stage3.py dexagent.yaml --name my_task  # runs/my_task/stage1 + stage2 -> runs/my_task/stage3
python stage3_trajectory_generation/run_stage3.py dexagent.yaml --semantic s.json --recon scene_recon.json --video clip.mp4
```

Needs: the model under `agent:` in `dexagent.yaml`, and a CPU.

Output, under `<run.output_dir>/<run.name>/stage3/`:

```
trajectory.npz            qpos [T, nq] of the Stage 2 scene (hands, fingers, arms, objects), fps, labels ('<side>: <phase>' per frame), subgoal (Stage 1 index per frame)
phases.json               every phase with its hand, objects, anchor and frame range
sim_video.mp4             the episode rendered from the video camera
stage4_task/              the folder Stage 4 runs on: stage4.json, scene.xml + assets/, trajectory.npz, human_video.mp4, sim_video.mp4
trajectory_manifest.json  per subgoal: motion, hand, object, pose and its source, passing stress tests, plan entry, expectation and measured outcome
episode/                  one subgoal_XX/ per subgoal: keyframes, physics rollout, outcome
trace.jsonl               every model turn, tool call and verifier verdict
new_tools.json            skills or verifiers the agent added to the library during the run, if any
```

Stage 4 then runs directly on the task folder:

```bash
python stage4_robot_data_generation/run_stage4.py runs/<name>/stage3/stage4_task/stage4.json
```
