import subprocess, sys
raise SystemExit(subprocess.run([sys.executable, "src/monitoring.py"]).returncode)
