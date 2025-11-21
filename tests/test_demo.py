import subprocess
import sys

def test_demo_runs():
    # Run train.py in demo mode with 1 epoch to validate it executes without crashing.
    cmd = [sys.executable, 'src\train.py', '--demo', '--epochs', '1', '--batch-size', '4', '--model-out', 'model_demo.h5']
    print('Running:', ' '.join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print('stdout:\n', proc.stdout)
    print('stderr:\n', proc.stderr)
    # We expect the process to exit with 0 even if it skips training due to missing TF.
    assert proc.returncode == 0
