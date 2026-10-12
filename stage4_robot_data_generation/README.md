# Stage 4: Robot Data Generation

Stage 3's verified robot trajectory and the human video in; randomized, photorealistic robot episodes out, each with its joint trajectory, ready for policy training.

| Step | What it does | Runs on |
|---|---|---|
| 1. `randomize` | Draws new initial object and robot states, adapts the trajectory to each one, and keeps only the episodes that pass the feasibility checks and the physics replay. | CPU |
| 2. `plate` | Removes the demonstrator's hands and the manipulated objects from the human video with SAM 3 and [VOID](https://github.com/netflix/void-model), giving a clean background plate. | GPU |
| 3. `export` | Writes a Blender scene for every episode and camera view: geometry, materials, per-frame poses, cameras. | CPU |
| 4. `render` | Renders the robot and the objects with Blender Cycles, from the calibrated ego camera and from any extra views such as wrist cameras. | GPU |
| 5. `composite` | Pastes the renders onto the clean plate and encodes one video per view. | CPU |

## Requirements

- **OS**: Linux, with an NVIDIA GPU and a recent driver.
- **GPU memory**: **40 GB or more** for the `plate` step. VOID loads a 5B-parameter video model and peaked at 38 GB in our runs on an RTX A6000 (48 GB). The `render` step needs about 8 GB.
- **Disk**: about 35 GB for model weights and 9 GB for the Python environment.
- **Python**: 3.11 exactly. The Blender wheel only exists for 3.11.
- **Blender**: you do **not** need to install Blender. It comes as the `bpy` Python package and is installed by `setup.sh`.

## Installation

```bash
conda create -y -n dexagent-stage4 python=3.11
conda activate dexagent-stage4
cd stage4_robot_data_generation
bash setup.sh
```

`setup.sh` does three things:

1. clones VOID into `third_party/void-model`,
2. installs the Python packages (VOID's pinned requirements, SAM 3, MuJoCo, Blender),
3. downloads the model weights listed below into the Hugging Face cache.

### Models to download in advance

All three are needed by the `plate` step. `setup.sh` downloads them; the table is here so you can fetch them yourself on a machine without internet access, or check that they are in place.

| Model | Hugging Face repo | File | Size | Access |
|---|---|---|---|---|
| VOID pass 1 | [`netflix/void-model`](https://huggingface.co/netflix/void-model) | `void_pass1.safetensors` | 11 GB | open |
| CogVideoX-Fun (VOID's base model) | [`alibaba-pai/CogVideoX-Fun-V1.5-5b-InP`](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.5-5b-InP) | whole repo | 20 GB | open |
| SAM 3.1 | [`facebook/sam3.1`](https://huggingface.co/facebook/sam3.1) | `sam3.1_multiplex.pt`, `config.json` | 3.5 GB | **gated** |

SAM 3.1 is gated. Request access on its model page, then log in before running `setup.sh`:

```bash
hf auth login
```

To download by hand:

```bash
hf download netflix/void-model void_pass1.safetensors
hf download alibaba-pai/CogVideoX-Fun-V1.5-5b-InP
hf download facebook/sam3.1 sam3.1_multiplex.pt config.json
```

The weights go to the Hugging Face cache (`$HF_HOME`, default `~/.cache/huggingface`). Set `HF_HOME` if you want them somewhere else. If VOID is cloned elsewhere, point `VOID_DIR` at it.

## Inputs

Stage 3 writes this folder for you (`runs/<name>/stage3/stage4_task/`). To prepare one by hand, put everything for one task in one folder, with a config file next to the data:

```
my_task/
├── scene.xml          simulation scene (MuJoCo MJCF): the robot and the objects
├── assets/            meshes and textures the scene refers to
├── trajectory.npz     the verified Stage 3 trajectory (and the servo commands that produced it)
├── trajectory_manifest.json  the Stage 3 subgoals and what each one must achieve, for the physics replay
├── human_video.mp4    the original human video
├── sim_video.mp4      optional: the Stage 3 simulation video, used only in a comparison video
└── stage4.json        the config: cameras, what to randomize, how many samples
```

| Input | Where it goes | Notes |
|---|---|---|
| Simulation | `scene.xml` | The scene Stage 2 reconstructed, with the robot from Stage 3, as MJCF with its meshes and textures next to it. |
| Robot | inside `scene.xml` | Name the body of each end effector in `robot.end_effectors`. |
| Trajectory | `trajectory.npz` | `qpos` of shape `[T, nq]`, `fps`, and `labels`: one phase name per frame. Optional `reference` `[T, nq]`: the hand servo commands Stage 3 executed (fingers commanded past the object's surface, so they press); the physics replay uses them. |
| Subgoal expectations | `trajectory_manifest.json` | Stage 3's manifest: per subgoal its frame window and `expect` (rise, in hand, at target, released, supported on, untouched ...). The physics replay re-checks them on every new layout. |
| Human video | `human_video.mp4` | The camera must not move during the clip. |
| Ego camera | `cameras.ego` | Position, orientation and vertical field of view of the camera that filmed the human video, in the scene's world frame. |
| New views | `cameras.views` | Extra cameras, each attached to a body: for a wrist camera, its pose relative to the hand and its field of view. |
| Number of samples | `num_samples` | How many randomized episodes to keep. |

## Run

```bash
conda activate dexagent-stage4
cd stage4_robot_data_generation
python run_stage4.py /path/to/my_task/stage4.json
```

That runs all five steps. To run or re-run only some of them:

```bash
python run_stage4.py /path/to/my_task/stage4.json --steps randomize
python run_stage4.py /path/to/my_task/stage4.json --steps export,render,composite
```

## Outputs

Everything is written under `output_dir` (default `outputs/` next to the config; for a run of `run_dexagent.py` that is `runs/<name>/stage3/stage4_task/outputs/`):

```
outputs/
├── randomize.json               every layout that was tried, and why rejected ones failed
├── plate/
│   ├── plate.png                the clean background
│   └── preview.mp4              human video with the erase mask | VOID result
├── samples/
│   ├── original/                the un-randomized trajectory
│   │   ├── ego.mp4, wrist_left.mp4, wrist_right.mp4
│   │   ├── preview.mp4          all views side by side
│   │   ├── compare.mp4          human video | simulation | generated
│   │   ├── trajectory.npz       qpos, joint_names, labels, fps
│   │   └── sample.json          the layout and the check results
│   ├── 0000/                    first randomized episode, the same files (no compare.mp4) plus
│   │   └── physics_replay.npz   what the contact-physics replay of this layout did
│   └── ...
└── render/                      Blender scenes and raw frames (safe to delete afterwards)
```

## Runtime

Measured on one RTX A6000 for the shipped cup example: a 351-frame trajectory rendered at every second frame (176 frames per episode), three views at 1280×1024, 32 samples per pixel, 4 render processes.

| Step | Time |
|---|---|
| `randomize` | about 1.7 s per draw for the kinematic checks, plus the physics replay |
| physics replay | one replay of the episode for the original, plus one per kinematically accepted layout (10 s for the cup, 23 s for the 1,630-frame computer installation) |
| `plate` | 6 min, once per task (SAM 3 65 s, VOID 190 s); 3 to 7 min on the other tasks |
| `export` | 33 s per episode |
| `render` | 8 min per episode |
| `composite` | 44 s per episode |

The whole example, three randomized episodes plus the original, ran in 43 minutes. Rendering dominates, and it scales with the number of episodes, views and frames. Raise `render.stride` or lower `render.samples` to trade quality for time. Renders are cached: re-running `render` skips frames that are already on disk.

## Third-party components

Stage 4 builds on [VOID](https://github.com/netflix/void-model) with its [CogVideoX-Fun](https://github.com/aigc-apps/CogVideoX-Fun) base model for video object removal, [SAM 3](https://github.com/facebookresearch/sam3) for text-prompted masks, [MuJoCo](https://mujoco.org) for kinematics, and [Blender](https://www.blender.org) Cycles for rendering. Each has its own license and terms; please check them before use.
