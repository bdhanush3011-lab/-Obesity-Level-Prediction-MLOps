import subprocess, sys
for script in ["src/train_production.py", "src/register_model.py"]:
    result = subprocess.run([sys.executable, script])
    if result.returncode: raise SystemExit(result.returncode)
