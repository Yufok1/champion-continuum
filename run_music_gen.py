import json
import sys
import os
from pathlib import Path

# Ensure we can import continuum_music_forge
sys.path.append(os.getcwd())
from continuum_music_forge import generate_hf_space_song

space_id = "ACE-Step/Ace-Step-v1.5"
api_name = "/generation_wrapper"
prompt = "Indiana Jones meets Bro-culture: The Raider of the Lost Gym. A quest for the Golden Shaker Bottle in the Temple of Gains. He dodges rolling medicine balls and booby-trapped squat racks., Cinematic Trap, High-Energy Bro-Step, Funny Hip-Hop, 130 BPM, Heavy Bass, 60 seconds"

# Using triple quotes for payload_json to avoid escaping issues
payload_json = r'''{"api_name": "/generation_wrapper", "kwargs": {"selected_model": "acestep-v15-xl-turbo", "generation_mode": "custom", "simple_query_input": "Indiana Jones meets Bro-culture: The Raider of the Lost Gym...", "simple_vocal_language": "en", "param_4": "Indiana Jones meets Bro-culture: The Raider of the Lost Gym...", "param_5": "[Verse 1]\nI'm swapping the idol for a 10lb plate\nDodging the boulders, I'm never late\nTemple of Doom? Nah, Temple of Squat\nFound the forbidden pre-workout, it's hot\n\n[Hook]\nBrotology! Raider of the Lost Gainz\nIndy with the snapback, breaking the chains\nI don't need a map, I follow the pump\nOver the pit of cardio I jump!\n\n[Outro]\nIt belongs in a museum... of swole.\n*Whip crack sound*", "param_6": 0, "param_7": "", "param_8": "", "param_9": "en", "param_10": 8, "param_11": 7.0, "param_12": true, "param_13": "-1", "param_14": null, "param_15": 60.0, "param_16": 1, "param_17": null, "param_18": "", "param_19": 0.0, "param_20": -1.0, "param_21": "Fill the audio semantic mask based on the given conditions:", "param_22": 1.0, "param_23": "text2music", "param_24": false, "param_25": 0.0, "param_26": 1.0, "param_27": 3.0, "param_28": "ode", "param_29": "", "param_30": "mp3", "param_31": 0.85, "param_32": true, "param_33": 2.0, "param_34": 0, "param_35": 0.9, "param_36": "NO USER INPUT", "param_37": true, "param_38": true, "param_39": true, "param_41": false, "param_42": true, "param_43": false, "param_44": false, "param_45": 0.5, "param_46": 8, "param_47": "vocals", "param_48": [], "param_49": false}}'''

title = "Raiders of the Lost Gym"

try:
    print(f"Starting music generation for: {title}")
    result = generate_hf_space_song(
        space_id=space_id,
        prompt=prompt,
        payload_json=payload_json,
        api_name=api_name,
        title=title
    )
    print("Generation complete.")
    print("RESULT_JSON_START")
    print(json.dumps(result, indent=2))
    print("RESULT_JSON_END")
except Exception as e:
    print(json.dumps({"status": "error", "error": str(e)}))
    sys.exit(1)
