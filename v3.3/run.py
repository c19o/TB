"""Wrapper that fixes the Python import deadlock before running a script."""
import sys
class _F:
    def find_module(self, name, path=None):
        return None
sys.meta_path.insert(0, _F())

# Now run the actual script
if len(sys.argv) > 1:
    script = sys.argv[1]
    sys.argv = sys.argv[1:]  # shift argv
    with open(script) as f:
        code = f.read()
    exec(compile(code, script, 'exec'), {'__name__': '__main__', '__file__': script})
