"""Freeze the existing evaluator implementation and upstream inputs for a run."""
from __future__ import annotations

import json
from pathlib import Path

from tool_library.common.evidence import file_digest


class InputContract:
    def __init__(self, library, inputs=()):
        library = Path(library).resolve()
        repo = library.parent
        files = set()
        for directory in (repo / "agent", library / "verifiers", library / "skills", library / "common",
                          repo / "stage1_semantic_understanding", repo / "stage2_scene_reconstruction",
                          repo / "stage3_trajectory_generation", repo / "stage4_robot_data_generation"):
            files.update(directory.rglob("*.py"))
            files.update((directory / "prompts").glob("*.md"))
            if (directory / "schema.json").exists():
                files.add(directory / "schema.json")
        self.index_path = library / "index.json"
        entries = json.loads(self.index_path.read_text()).get("tools", []) if self.index_path.exists() else []
        self.tools = {entry["name"]: entry for entry in entries}
        if len(self.tools) != len(entries):
            raise ValueError("library index contains duplicate tool names")
        for entry in inputs:
            path = Path(entry).resolve()
            if path.is_dir():
                files.update(p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
            else:
                files.add(path)
        self.hashes = {str(p.resolve()): file_digest(p) for p in sorted(files)}
        # A reconstruction may reference meshes or includes outside its folder.
        # Freeze compiled content as well as direct files, so changing an external
        # asset cannot change both the candidate and its trusted reference.
        self.models = {}
        manifests = [p for p in files if p.name == "scene_recon.json"]
        if manifests:
            import mujoco
            from agent.artifacts import model_digest
            for path in manifests:
                entry = json.loads(path.read_text())
                scene = (path.parent / entry.get("scene_xml", "scene.xml")).resolve()
                self.models[str(scene)] = model_digest(mujoco.MjModel.from_xml_path(str(scene)))

    def check(self):
        issues = []
        for name, expected in self.hashes.items():
            try:
                current = file_digest(name)
            except OSError:
                current = None
            if current != expected:
                issues.append(f"trusted input or verification code changed during this run: {name}")
        try:
            entries = json.loads(self.index_path.read_text()).get("tools", []) if self.index_path.exists() else []
            tools = {entry["name"]: entry for entry in entries}
            if len(tools) != len(entries):
                issues.append("library index contains duplicate tool names")
            for name, entry in self.tools.items():
                if tools.get(name) != entry:
                    issues.append(f"existing trusted tool metadata changed during this run: {name}")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            issues.append(f"cannot validate the library index: {exc}")
        if self.models:
            import mujoco
            from agent.artifacts import model_digest
            for scene, expected in self.models.items():
                try:
                    current = model_digest(mujoco.MjModel.from_xml_path(scene))
                except Exception:
                    current = None
                if current != expected:
                    issues.append(f"trusted compiled scene or an external asset changed during this run: {scene}")
        return {"ok": not issues, "issues": issues, "files_checked": len(self.hashes)}

    def write(self, path):
        Path(path).write_text(json.dumps({"schema": "dexagent.input_contract.v1", "sha256": self.hashes,
                                         "compiled_models": self.models, "trusted_tools": self.tools}, indent=1))
