import subprocess, sys
raise SystemExit(subprocess.run([sys.executable, "src/failure_injection.py"]).returncode)
