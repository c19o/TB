"""Launch a Python script with import deadlock fix.
Usage: python pyfix.py <script.py> [args...]
"""
import sys, runpy
class _T:
    def find_module(self, n, p=None):
        sys.stdout.write('')  # tiny GIL release
        return None
sys.meta_path.insert(0, _T())

if len(sys.argv) < 2:
    print("Usage: python pyfix.py <script.py> [args...]")
    sys.exit(1)

script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name='__main__')
