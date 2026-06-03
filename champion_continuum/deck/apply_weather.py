import json
import urllib.request

url = "http://127.0.0.1:7866/api/tool/env_control"
payload = {
    "command": "weather_set_surface",
    "target_id_json": {
        "enabled": True,
        "source": "workbench_dance_weather_resource",
        "kind": "energy",
        "flow_class": "gravity_inversion_particle_accent",
        "density": 0.64,
        "speed": 0.92,
        "turbulence": 0.47,
        "direction": {"x": 0.18, "y": 0.34, "z": -0.92},
        "drift": {"x": 0.06, "y": 0.18, "z": -0.12},
        "glyphs": "PHOENIXWINDMANA",
        "glyph_stride": 4,
        "render_fidelity": "adaptive",
        "color_hint": "#facc15",
        "accent_color_hint": "#60a5fa",
        "weather_layers": [
            {
                "lane_index": 1,
                "kind": "energy",
                "flow_class": "everquest_particle_burst",
                "density": 0.72,
                "speed": 1.06,
                "turbulence": 0.58,
                "glyphs": "MANA",
                "glyph_stride": 4,
                "color_hint": "#a78bfa",
                "accent_color_hint": "#f0abfc"
            },
            {
                "lane_index": 2,
                "kind": "mist",
                "flow_class": "gravity_soft_lift",
                "density": 0.42,
                "speed": 0.7,
                "turbulence": 0.25,
                "glyphs": "LIFT",
                "glyph_stride": 4,
                "color_hint": "#67e8f9",
                "accent_color_hint": "#fde68a"
            }
        ]
    },
    "actor": "Gemini"
}

data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
with urllib.request.urlopen(req) as response:
    print(response.read().decode("utf-8"))
