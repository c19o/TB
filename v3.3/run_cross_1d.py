import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ['V2_RIGHT_CHUNK'] = '500'

class _F:
    def find_module(self, n, p=None):
        try: os.write(1, b'.')
        except: pass
        return None
sys.meta_path.insert(0, _F())

sys.argv = ['v2_cross_generator.py', '--tf', '1d', '--symbol', 'BTC', '--save-sparse']
import runpy
runpy.run_path('v2_cross_generator.py', run_name='__main__')
