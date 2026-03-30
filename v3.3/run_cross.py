"""Run v2_cross_generator.py with import deadlock fix."""
import sys, os

# Fix: os.write(1) releases GIL during imports, preventing deadlock
class _F:
    def find_module(self, n, p=None):
        try:
            os.write(1, b'.')
        except:
            pass
        return None
sys.meta_path.insert(0, _F())

# Now import and run the cross generator
import runpy
sys.argv[0] = 'v2_cross_generator.py'
runpy.run_path('v2_cross_generator.py', run_name='__main__')
