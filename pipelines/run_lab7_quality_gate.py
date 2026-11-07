import subprocess, sys
for script in ["src/quality_gate.py"]:
    result = subprocess.run([sys.executable, script])
    if result.returncode: raise SystemExit(result.returncode)
