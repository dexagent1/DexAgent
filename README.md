<div align="center">

<h1>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/title_dark.png">
    <img src="assets/title_light.png" alt="DexAgent" height="80">
  </picture>
  <br>
  <sub>An Agentic Human2Sim2Robot Framework for Dexterous Manipulation with Self-Evolving Tool Library</sub>
</h1>

</div>

## Library

DexAgent keeps every skill and verifier it develops in a self-evolving tool library. The **[library visualizer](https://dexagent1.github.io/library/)** shows the library live: how many skills and verifiers there are, each one by category, and its inputs, outputs and source. The tools themselves live in [`tool_library/`](tool_library/); its README explains how to [contribute a tool to the library](tool_library/README.md#contributing-to-the-library) and how to [use an individual tool by hand](tool_library/README.md#using-a-tool-by-hand).

## Quick start

### 0. ⚠️ Cost and run time

Each agent stage is a long loop of model calls. What a run costs, and how long it takes, is not known in advance.

- **API key:** before the first run, set `max_rounds` for every stage and `agent.budget_usd` (the run's spending cap) in `dexagent.yaml`; DexAgent stops a stage when either is spent.
- **Subscription:** to avoid API charges, you can ask your coding agent (e.g., Claude Code or Codex) to fork a subagent on the model you choose and run the pipeline through the tool-call CLI ([running a stage by hand](agent/README.md#running-a-stage-by-hand)). Give that subagent a turn limit as well, so it cannot use up your token allowance.
- **Run time** depends on the video length, the model, and how much of the task the tool library already covers: when it lacks the tools a task needs, the agent writes and tests new ones, which takes longer.
- **Model:** we recommend Claude Opus 5 or GPT-5.6, but the best choice depends on the task.

### 1. Install

```bash
git clone https://github.com/dexagent1/DexAgent.git && cd DexAgent
conda create -y -n dexagent python=3.11 && conda activate dexagent
pip install -r requirements.txt
```

This is the harness and every CPU tool: Stages 1 and 3 run on it with the language model named in `dexagent.yaml`, and Stage 2 adds SAM 3 for the object masks (its own environment, a GPU; first row below). The physics is validated on MuJoCo 3.10 with NumPy 1.x, the versions `requirements.txt` pins; other versions simulate contacts differently. Optional parts live in their own environments and are named in `dexagent.yaml`:

| Part | Needed for | Install |
|---|---|---|
| SAM 3, SAM 2, HaMeR, Video Depth Anything, GeoCalib, SAM 3D | the perception skills (masks, tracks, depth, gravity, hand meshes, mesh objects); a GPU | `environments:` in `dexagent.yaml` |
| Isaac Sim 5.0 | Stage 2 on deformable cords | `environments.isaac` |
| BODex, DexGraspNet | the optional GPU grasp priors | `models.bodex_dir`, `models.dexgraspnet_dir` |
| VOID, Blender (`bpy`) | Stage 4; a GPU with 40 GB | [Stage 4 README](stage4_robot_data_generation/README.md) (`setup.sh`, Python 3.11), then `environments.stage4` |

A tool whose package is missing says so when it is called; the agent then uses another one.

### 2. Setup

Copy [`dexagent.yaml`](dexagent.yaml) and fill in your model API key and model name, your video, the camera that filmed it, and your robot setup. **Anything left empty falls back to the default: our bench in [`setup/`](setup/), its calibrated ego camera, and the weights in the Hugging Face cache.** To use your own robot or bench, copy `setup/` and edit it as its README describes.

### 3. Run

One command runs the four stages in order and prints a closing table; each stage also runs on its own (its README says what it reads, what it writes and which GPU it needs).

```bash
python run_dexagent.py dexagent.yaml                                  # video + prompt -> randomized robot videos, stages 1 -> 4
python run_dexagent.py dexagent.yaml --from stage3                    # resume from a stage; --to stage2 stops after one

python stage1_semantic_understanding/run_stage1.py dexagent.yaml      # video + prompt -> scene_semantics.json
python stage2_scene_reconstruction/run_stage2.py dexagent.yaml        # + video -> scene.xml with the verified, placed objects (Isaac Sim for deformables)
python stage3_trajectory_generation/run_stage3.py dexagent.yaml       # + subgoals -> one robot episode verified in physics, packaged for Stage 4
python stage4_robot_data_generation/run_stage4.py runs/<name>/stage3/stage4_task/stage4.json   # -> randomized, retextured robot videos in stage4_task/outputs/
```

## Code

| Part | Folder |
|---|---|
| Agent harness | [`agent/`](agent/) |
| Stage 1: Semantic Understanding | [`stage1_semantic_understanding/`](stage1_semantic_understanding/) |
| Stage 2: Property-Based Simulation Reconstruction | [`stage2_scene_reconstruction/`](stage2_scene_reconstruction/) |
| Stage 3: Robot Trajectory Generation | [`stage3_trajectory_generation/`](stage3_trajectory_generation/) |
| Stage 4: Robot Data Generation | [`stage4_robot_data_generation/`](stage4_robot_data_generation/) |
| Tool library | [`tool_library/`](tool_library/) |
| Default setup | [`setup/`](setup/) |
| Library visualizer | [`library_visualizer/`](library_visualizer/) |

## BibTeX

If you find DexAgent useful in your research, please cite:

```bibtex
@article{wang2026dexagent,
  title   = {DexAgent: An Agentic Human2Sim2Robot Framework for Dexterous
             Manipulation with Self-Evolving Tool Library},
  author  = {Wang, Youhui and Li, Yunzhu and Fei-Fei, Li and Wu, Jiajun and
             Huang, Huang},
  journal = {arXiv preprint arXiv:2609.35318},
  year    = {2026}
}
```
