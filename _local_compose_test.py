"""Local, GPU-free test of the Stage + memory-awareness fix.

Stubs the HF `spaces` package (which only exists in the HF runtime; on CPU its
GPU decorator is effectively a pass-through) so we can import app.py and drive the
REAL functions with a simulated model reply -- no model, no GPU, no spend.
"""
import sys, types, warnings
warnings.filterwarnings("ignore")

# --- stub `spaces` so the GPU decorator is a no-op pass-through (CPU behaviour) ---
spaces = types.ModuleType("spaces")
def GPU(*a, **k):
    def deco(fn):
        return fn
    # support both @spaces.GPU and @spaces.GPU(duration=...)
    if a and callable(a[0]):
        return a[0]
    return deco
spaces.GPU = GPU
sys.modules["spaces"] = spaces

import app

print("IMPORT OK\n")

# 1) Plain-lane prompt now carries memory identity + compose grammar?
p = app._plain_system_prompt({"mode": "TALK"})
print("TALK prompt teaches memory :", "persistent Continuum memory" in p)
print("TALK prompt teaches compose:", "[[compose:" in p)
pp = app._plain_system_prompt({"mode": "PLAN"})
print("PLAN prompt teaches compose:", "[[compose:" in pp)

# 2) The end-to-end render path a plain-lane answer now runs through.
state = app._new_session()
fake_reply = (
    "Yes -- my memory is fully operational; I have persistent Continuum memory and "
    "a live trace facility.\n\n"
    "[[compose: {\"blocks\":["
    "{\"type\":\"heading\",\"tone\":\"scarab\",\"text\":\"Memory Status\"},"
    "{\"type\":\"kv\",\"tone\":\"success\",\"items\":[{\"k\":\"memory\",\"v\":\"operational\"},{\"k\":\"nodes\",\"v\":\"live\"}]},"
    "{\"type\":\"badges\",\"tone\":\"insight\",\"labels\":[\"continuum\",\"merkle-chained\"]}"
    "]}]]"
)
prose, stage = app._apply_composition(fake_reply, state)
print("\n--- _apply_composition on a simulated reply ---")
print("prose shown to user :", repr(prose))
print("directive stripped  :", "[[compose:" not in prose)
print("stage got HTML      :", isinstance(stage, str) and "cc-stage" in stage)
print("stage has 'operational':", isinstance(stage, str) and "operational" in stage)
print("stage XSS-safe (no <script>):", isinstance(stage, str) and "<script" not in stage)

# 3) Did composing record a memory node (informational wealth)?
kinds = [s.get("kind") for s in state["trace"].breadcrumb(50)]
print("compose receipt recorded:", "compose" in kinds)

# 4) A plain answer with NO directive must leave the stage untouched (gr.update()).
prose2, stage2 = app._apply_composition("Just a normal answer, no panel.", state)
import gradio as gr
print("no-directive -> stage unchanged:", type(stage2).__name__ == "EventData" or stage2.__class__.__name__ == "dict" or "update" in str(type(stage2)).lower() or stage2 == gr.update())
print("\nALL CHECKS DONE")
