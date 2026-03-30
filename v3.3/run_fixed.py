"""Run any Python script with the Windows import deadlock fix.

Usage: python -u run_fixed.py <script.py> [args...]

Fixes a GIL/import lock deadlock on Windows with numpy 2.2+/pandas 3.0+/sklearn 1.8+
by inserting a meta_path finder that does os.write(1) to release the GIL during imports.
"""
import sys, os

class _GILReleaseFinder:
    def find_module(self, name, path=None):
        try:
            os.write(1, b'.')
        except Exception:
            pass
        return None

sys.meta_path.insert(0, _GILReleaseFinder())

if len(sys.argv) < 2:
    print("Usage: python -u run_fixed.py <script.py> [args...]")
    sys.exit(1)

# Set up for the target script
script = sys.argv[1]
sys.argv = sys.argv[1:]

import runpy
runpy.run_path(script, run_name='__main__')
