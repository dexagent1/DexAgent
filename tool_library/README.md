# Tool library

The self-evolving tool library of DexAgent: the **skills** that produce assets, poses and motions, and the **verifiers** that check them. Every tool is one Python file, so it can be read, run and reused on its own. The agent adds a tool when a capability is missing and keeps it only after its test passes.

Browse the library in the [library visualizer](https://dexagent1.github.io/library/).

```
tool_library/
├── skills/         one file per skill
├── verifiers/      one file per verifier
├── common/         shared helpers the tools import (MJCF writing, hand rig, trajectories, video, rendering)
├── tests/          test scripts; every tool is exercised by one of them
├── build_index.py  reads the metadata of every tool into index.json and TASKS.md
├── index.json      what the visualizer shows
├── TASKS.md        which tools each of the paper's eleven tasks used
└── CONVENTIONS.md  how a tool file is written
```

## Contributing to the library

The library is meant to grow with its users. A tool reaches the shared library through a pull request, whether your agent created it during a run or you wrote it by hand.

### 1. Fork and branch

Fork [dexagent1/DexAgent](https://github.com/dexagent1/DexAgent) on GitHub, clone your fork, and create a branch named after the tool:

```bash
git clone https://github.com/<you>/DexAgent.git && cd DexAgent
git checkout -b add-<tool_name>
```

### 2. Add the tool

**A tool your agent created.** `library_create` writes it into your `tool_library/` (`skills/<name>.py` or `verifiers/<name>.py`, with its test in `tests/test_<name>.py`) and lists it in the run's `runs/<name>/<stage>/new_tools.json`. It has already passed its test on at least three variations, so it only needs the checks of step 3.

**A tool you write by hand.**

1. Create `skills/<name>.py` or `verifiers/<name>.py`. Use a name that says what the tool does, such as `penetration_check`.
2. Start the file with a module-level `TOOL = {...}` dictionary. It must be a plain literal, because the index is built without importing the file:

   | Key | Content |
   |---|---|
   | `name`, `title` | identifier (matches the file name) and display name |
   | `type` | `skill` or `verifier` |
   | `category` | one of the categories in `CATEGORIES` in `build_index.py` |
   | `summary` | one sentence |
   | `description` | what it does, how it decides, what it assumes |
   | `inputs`, `outputs` | lists of `{name, type, description}` |
   | `status` | `tested` or `ported`, see CONVENTIONS.md |
   | `entrypoint` | the function the agent calls |
   | `requirements`, `simulators`, `tags`, `used_in`, `cli` | optional |

3. Implement the tool below the metadata. A verifier returns a report dictionary with an `ok` field; a skill returns what it produced. Give each a command line entry point when that makes sense.
4. Write `tests/test_<name>.py`: run the tool on at least three distinct cases, print `[PASS] <case>` for each, and exit non-zero on a failure. A verifier needs a case that must pass and one that must fail.

See [`verifiers/penetration_check.py`](verifiers/penetration_check.py) for a complete example and [CONVENTIONS.md](CONVENTIONS.md) for the code rules.

Contributed tools follow the same rules as the ones the agent creates:

- **Do not change an existing tool's behaviour.** Runs and examples depend on it. Fork it under a new name instead; fix a bug in place only together with a test that fails before the fix.
- **Generalize.** A tool is a parameterized function for a class of objects or motions, not for one video: no hard-coded frames, pixel boxes, object names or file paths.
- **Search first.** Check the [library visualizer](https://dexagent1.github.io/library/) for a tool that already covers the capability, so the library does not collect duplicates.

### 3. Test before you push

```bash
python tool_library/build_index.py                      # refuses malformed metadata; rewrites index.json and TASKS.md
MUJOCO_GL=egl python tool_library/tests/test_<name>.py  # the tool's own test
python tool_library/tests/run_all.py <topic>            # the test scripts whose name contains <topic>, e.g. grasp, stage3
```

Run the full suite (`python tool_library/tests/run_all.py`, CPU, about an hour on 4 cores) when you change anything in `common/`. It needs no API key; tests of a model-backed tool skip when its package is absent.

### 4. Open a pull request

Commit the tool, its test, `index.json` and `TASKS.md` together, push the branch to your fork, and open a pull request against `main`. Keep one tool, or one small family of related tools, per pull request. In the description, say what capability the tool adds, which task or video needed it, and paste the output of its test.

## Using a tool by hand

Every tool can be used outside an agent run, in three ways.

**As a Python function** with keyword arguments:

```python
from tool_library.skills.construct_bottle import construct_bottle
from tool_library.verifiers.thread_self_locking_test import thread_self_locking_test

built = construct_bottle("out/bottle", body_radius=0.03, body_height=0.12, thread_pitch=0.004,
                         thread_turns=4, body_rgba=[0.9, 0.9, 0.9, 0.35])
report = thread_self_locking_test(built)
assert report["ok"]
```

**From the command line, exactly as the agent calls it.** `agent.toolcall` takes the tool name and its arguments as JSON (or `@file.json`) and prints the JSON result as the last line on stdout. A tool whose packages live in another environment is run in the interpreter named under `environments:` in `dexagent.yaml` (`--config` for another file).

```bash
python -m agent.toolcall make_contact_sheet '{"video": "clip.mp4", "out_path": "sheet.jpg"}'
python -m agent.toolcall scene_semantic_schema_check @args.json
```

**Through its own command line**, for tools that have one (the `cli` field of its metadata, shown in the visualizer):

```bash
python -m tool_library.verifiers.penetration_check scene.xml trajectory.npz --baseline reference.npz
```

Tools that wrap a neural model (SAM 3, SAM 2, depth, HaMeR, SAM 3D, VOID) or a GPU optimizer (BODex, DexGraspNet) import it inside the function and name the package when it is missing; their requirements are listed in their metadata.
