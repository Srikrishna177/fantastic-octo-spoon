#!/usr/bin/env python
"""Debug test for 04_stats.py"""

import sys
import traceback
from pathlib import Path

# Load the 04_stats module
sys.path.insert(0, 'pipeline')
import importlib.util

spec = importlib.util.spec_from_file_location("stats_module", "pipeline/04_stats.py")
stats_module = importlib.util.module_from_spec(spec)

try:
    print("[DEBUG] Loading module...")
    spec.loader.exec_module(stats_module)
    print("[DEBUG] Module loaded successfully")
    
    print("[DEBUG] Calling compute_stats video_ids=['vid_001']...")
    result = stats_module.compute_stats(['vid_001'])
    print(f"[DEBUG] compute_stats returned: {type(result)}")
    
    # Check if output directory was created
    vid_001_dir = Path("outputs/vid_001")
    print(f"[DEBUG] Checking if outputs/vid_001 exists: {vid_001_dir.exists()}")
    
    if vid_001_dir.exists():
        print(f"[DEBUG] Files in {vid_001_dir}:")
        for f in vid_001_dir.iterdir():
            print(f"  - {f.name} ({f.stat().st_size} bytes)")
    else:
        print("[DEBUG] outputs/vid_001 does NOT exist!")
        
except Exception as e:
    print(f"[ERROR] {type(e).__name__}: {e}")
    traceback.print_exc()
