import subprocess, sys
raise SystemExit(subprocess.run([sys.executable, "src/retraining.py"]).returncode)
