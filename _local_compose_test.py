"""Manual real smoke check for the app composition path.

This script intentionally does not patch or replace HF/Gradio dependencies. It
imports the actual app and runs the real composition renderer with a deterministic
directive payload. If the local environment cannot initialize real Gradio/OAuth,
the script exits with the real import error instead of simulating success.
"""

import sys


try:
    import app
except Exception as exc:
    print("REAL APP IMPORT FAILED")
    print(f"{type(exc).__name__}: {exc}")
    raise


def main() -> int:
    print("REAL APP IMPORT OK")

    prompt = app._plain_system_prompt({"mode": "TALK"})
    checks = {
        "talk_prompt_mentions_memory": "persistent Continuum memory" in prompt,
        "talk_prompt_mentions_compose": "[[compose:" in prompt,
    }

    state = app._new_session()
    directive_reply = (
        "Composition smoke check.\n\n"
        "[[compose: {\"blocks\":["
        "{\"type\":\"heading\",\"tone\":\"scarab\",\"text\":\"Memory Status\"},"
        "{\"type\":\"kv\",\"tone\":\"success\",\"items\":[{\"k\":\"memory\",\"v\":\"operational\"}]}"
        "]}]]"
    )
    prose, stage = app._apply_composition(directive_reply, state)
    checks.update(
        {
            "directive_stripped": "[[compose:" not in prose,
            "stage_rendered": isinstance(stage, str) and "cc-stage" in stage,
            "stage_contains_value": isinstance(stage, str) and "operational" in stage,
            "stage_blocks_script": isinstance(stage, str) and "<script" not in stage.lower(),
        }
    )

    for name, ok in checks.items():
        print(f"{name}: {ok}")

    if not all(checks.values()):
        return 1
    print("REAL COMPOSE SMOKE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
