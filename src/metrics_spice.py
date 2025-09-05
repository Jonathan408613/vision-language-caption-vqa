"""
Thin wrapper to run SPICE if Java and SPICE jar exist.
Place SPICE jar at tools/spice/SPICE-1.0.jar
Download from: https://github.com/peteanderson80/SPICE/releases (or build)
"""
import json, os, subprocess, tempfile
from typing import List

def compute_spice(gts_json: str, res_json: str, spice_jar: str = "tools/spice/SPICE-1.0.jar"):
    if not os.path.exists(spice_jar):
        print("SPICE jar not found; skipping SPICE.")
        return None
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "spice_out.json")
        cmd = [
            "java", "-Xmx8G", "-jar", spice_jar,
            gts_json, res_json, "-out", out, "-subset",
        ]
        subprocess.run(cmd, check=True)
        with open(out) as f:
            data = json.load(f)
        # Overall score
        scores = [d.get("SPICEScore", 0.0) for d in data]
        return sum(scores)/len(scores) if scores else None