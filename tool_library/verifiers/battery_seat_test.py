"""Battery seat test: a seated cell stays, a released cell is thrown out.

Attaches a `construct_battery` cell to a `construct_battery_case` slot and
runs the case's own physics receipt. Seated against the spring plate with
the button end behind the retaining bump, the cell must not drift for 1.5 s.
Pressed 2 mm deeper with the button end tilted 25 degrees up, the way a hand
holds it mid-insertion, and then released, the spring must re-extend fully
and throw the cell back down the slot. Nothing is actuated: gravity, the
spring joints and contact do all of it.

    from tool_library.verifiers.battery_seat_test import test_battery_seat
    r = test_battery_seat(case_built, battery_built, slot=0)
"""
from __future__ import annotations

TOOL = {
    "name": "battery_seat_test",
    "title": "Battery seat test",
    "type": "verifier",
    "category": "Reconstruction Mechanism Tests",
    "summary": "Seated battery must stay in its sprung slot; a battery released mid-insertion "
               "must be ejected by the spring.",
    "description": (
        "Builds a combined model of a battery case slot and a cell (the cell's XML is attached "
        "under the prefix 'bat_'), then runs two trials from posed initial states with no "
        "forces applied afterwards. Seated: the cell is placed on the compressed spring plate "
        "behind the bump; after 1.5 s its drift must be under 1 mm and the spring must still "
        "hold more than half its seated compression. Eject: the cell is placed 2 mm deeper "
        "with the button end tilted 25 degrees up, clear of the bump, and released; after "
        "0.8 s the spring must have re-extended (under 1 mm left) and the cell must have "
        "travelled more than 6 mm down the slot. Both must hold for the slot to pass. This "
        "is how the case's bump height, spring preload and slot clearance are checked to be a "
        "mechanism rather than a cosmetic recess."
    ),
    "inputs": [
        {"name": "case_built", "type": "dict | str",
         "description": "The dict construct_battery_case returned, or the path of its <name>_params.json."},
        {"name": "battery_built", "type": "dict | str",
         "description": "The dict construct_battery returned, or the path of its <name>_params.json."},
        {"name": "slot", "type": "int", "description": "Which slot to test. Default 0."},
        {"name": "out_json", "type": "str | None", "description": "Write the report here."},
    ],
    "outputs": [
        {"name": "ok", "type": "bool", "description": "Both the seated and the eject trials passed."},
        {"name": "verdict", "type": "str", "description": "One sentence with the drift and the ejection distance."},
        {"name": "measured", "type": "dict",
         "description": "slot, seated {drift_mm, spring_holds_mm, ok}, eject {thrown_mm, spring_after_mm, ok}."},
    ],
    "status": "tested",
    "entrypoint": "battery_seat_test",
    "requirements": ["mujoco", "numpy", "scipy"],
    "simulators": ["mujoco"],
    "tags": ["mechanism", "battery", "spring", "insertion", "seat"],
    "used_in": ["Battery Insertion"],
    "cli": "python -m tool_library.verifiers.battery_seat_test <case_params.json> <battery_params.json> [slot]",
}


def test_battery_seat(case_built, battery_built, slot: int = 0,
                      out_json=None) -> dict:
    from tool_library.common import battery_case as BC
    from tool_library.common import battery_cell as BAT
    from tool_library.common.mechtest import resolve_built, report

    case_built = resolve_built(case_built, BC.rebuild)
    battery_built = resolve_built(battery_built, BAT.rebuild)
    meas = BC.probe(case_built, battery_built, slot=int(slot))
    s, e = meas["seated"], meas["eject"]
    verdict = (f"SEAT OK — seated cell drifted {s['drift_mm']:.2f} mm in 1.5 s with the "
               f"spring holding {s['spring_holds_mm']:.1f} mm; released mid-insertion it was "
               f"thrown {e['thrown_mm']:.1f} mm down the slot and the spring re-extended to "
               f"{e['spring_after_mm']:.2f} mm"
               if meas["ok"] else
               f"SEAT FAILED — seated {'ok' if s['ok'] else 'DRIFTED ' + str(s['drift_mm']) + ' mm'}, "
               f"eject {'ok' if e['ok'] else 'only ' + str(e['thrown_mm']) + ' mm, spring left at ' + str(e['spring_after_mm']) + ' mm'}")
    return report(bool(meas["ok"]), verdict, meas, out_json=out_json)


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(test_battery_seat(sys.argv[1], sys.argv[2],
                                       int(sys.argv[3]) if len(sys.argv) > 3 else 0),
                     indent=1))


battery_seat_test = test_battery_seat   # the library calls every tool by its own name
