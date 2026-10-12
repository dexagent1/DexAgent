# Writing a tool

Every skill and verifier in this library is **one Python file** that works on its own, with `tool_library/common/` as its only internal dependency. Nothing here imports from the research code the tools were extracted from. The exceptions are tools that check or package a whole stage of the DexAgent pipeline and so import the code around the library: `agent/` (`stage2_gate_suite` uses `agent.reference_evidence`) and Stage 4's `stage4` package (`stage4_handoff_check` loads `stage4.json` with Stage 4's own config loader).

## File layout

```
tool_library/
├── skills/<name>.py        one skill per file
├── verifiers/<name>.py     one verifier per file
├── common/<topic>.py       shared helpers (MJCF writing, geometry, video)
├── tests/test_<topic>.py   plain-python smoke tests, one per topic
├── build_index.py          TOOL metadata -> index.json
└── index.json
```

## Names

`snake_case`, saying what the tool does. Skills are verbs on nouns: `construct_bottle`, `measure_object_trajectory`, `pose_from_points`. Verifiers end in `_check`, `_test`, `_audit` or `_gate`: `penetration_check`, `thread_self_locking_test`, `object_teleport_audit`, `stacked_on_gate` (the stage suites end in `_gate_suite`, and the one reachability map, `arm_workspace_map`, in `_map`); a variant of a verifier adds a qualifier after the suffix (`grasp_stress_test_preopen`). The file name is the tool name.

## The TOOL literal

The file starts with a module-level `TOOL = {...}` **literal**; only a module docstring may precede it (no other expressions, no imports before it). `build_index.py` reads it with `ast.literal_eval`.

| Key | Required | Content |
|---|---|---|
| `name`, `title` | yes | identifier = file name; a short display name |
| `type` | yes | `skill` or `verifier` |
| `category` | yes | one of the six categories for the tool's type in `CATEGORIES` in `build_index.py` |
| `summary` | yes | one sentence |
| `description` | yes | what it does, how it decides, what it assumes; several sentences |
| `inputs`, `outputs` | yes | lists of `{"name", "type", "description"}`; types are plain words, units stated |
| `status` | yes | `tested`: the smoke test ran here and passed; `ported`: runs but needs an external model or hardware this machine could not exercise end to end |
| `entrypoint` | yes | the function name |
| `requirements` | no | pip names, with versions where it matters |
| `simulators` | no | `MuJoCo`, `Isaac Sim`, ... |
| `tags`, `used_in`, `cli` | no | `used_in` names the tasks that used it: the paper's eleven by their names, any other task by a short name of its own (TASKS.md lists those under "Other tasks") |

## Code rules

- The entry point is an ordinary function with keyword arguments and sensible defaults. Paths in, paths or plain dictionaries out. A verifier returns a dictionary with an `ok` boolean and the numbers it decided on.
- Parametric: a constructor takes dimensions, colors and counts as arguments and writes an MJCF (or URDF) plus its meshes into an output folder it is given. Nothing is hard-coded to one sample.
- No reads from a hidden workspace, no global state, no absolute paths to this machine. Assets a tool needs (a robot hand, a bench) are arguments.
- Heavy dependencies (a neural model, a GPU optimizer) are imported inside the function, and a missing one raises an error that names the package and how to get it.
- Every tool has a smoke test in `tests/`. A constructor compiles its output in MuJoCo and asserts the mechanism moves; a verifier is run on a case that must pass and one that must fail.
- Comments explain decisions and measured thresholds, not syntax.

## Adding the tool to the library

```bash
python tool_library/build_index.py          # refuses malformed metadata
python tool_library/tests/test_<topic>.py   # must pass
```

Commit the tool, the test and `index.json` together.
