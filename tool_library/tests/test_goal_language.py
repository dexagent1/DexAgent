"""Equivalent orientation/release language must enforce the same physical goal."""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from tool_library.common import task_gates as G


def main():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'scene.xml'
        path.write_text('<mujoco><worldbody><geom name="floor" type="plane" size="1 1 .1"/>'
                        '<body name="vessel" pos="0 0 .02"><freejoint/><geom type="box" size=".02 .02 .02" mass=".1"/></body>'
                        '</worldbody></mujoco>')
        model = mujoco.MjModel.from_xml_path(str(path))
        for wording in ('inverted', 'upside-down', 'mouth-down'):
            sem = {'objects': [{'id': 'vessel', 'name': 'vessel', 'type': 'rigid',
                                'initial_state': f'Resting {wording} on the table.'}],
                   'task_success': f'The vessel rests {wording} on the table.',
                   'subgoals': [{'index': 1, 'acting_hand': 'left', 'motion': 'place',
                                'object_refs': ['vessel'], 'goal': f'The vessel rests {wording} on the table.'}]}
            man = {'subgoals': [{'index': 1, 'stage1_indices': [1], 'object': 'vessel',
                                 'side': 'left', 'motion': 'place', 'frames': [10, 39]}]}
            recon = {'objects': [{'id': 'vessel', 'root_body': 'vessel'}]}
            for changed in (False, True):
                Q = np.repeat(model.qpos0[None], 40, axis=0)
                if changed:
                    Q[5:, 3:7] = [0., 1., 0., 0.]
                episode = G.Episode(model, qpos=Q, fps=30., labels=['left: place #1'] * len(Q))
                report = G.stage1_goal_report(episode, sem, man, recon)
                assert report['ok'] is (not changed), (wording, changed, report)
            print(f'[PASS] {wording}: preserving orientation passes and flipping it fails')
        episode = G.Episode(model, qpos=Q, fps=30., labels=['left: place #1'] * len(Q))
        gc = G.GoalContext(episode, sem, recon)
        for phrase in ('when both hands withdraw', 'after the hands withdraw', 'with no hand contact'):
            goal = {'index': 1, 'acting_hand': 'both', 'motion': 'other', 'object_refs': ['vessel'],
                    'goal': f'The vessel remains stable {phrase}.'}
            predicates, _ = G.infer_goal_predicates(gc, goal, man['subgoals'])
            assert any(p['class'] == 'release_settle' for p in predicates), (phrase, predicates)
        print('[PASS] withdrawal phrases require independently measured release and settling')


if __name__ == '__main__':
    main()
