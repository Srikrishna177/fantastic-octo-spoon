import importlib.util
import sys

spec = importlib.util.spec_from_file_location('stats_module', 'pipeline/04_stats.py')
stats_module = importlib.util.module_from_spec(spec)

try:
    spec.loader.exec_module(stats_module)
    print("Module loaded")
except Exception as e:
    print(f"[ERROR during module load] {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("Module attributes:", [x for x in dir(stats_module) if not x.startswith('_')])
