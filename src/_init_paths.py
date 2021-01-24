import os.path as osp
import sys

try:
    import dcn_v2
except ImportError:
    import pip
    pip.main(['install', '-e','git+https://github.com/CharlesShang/DCNv2@c7f778f28b84c66d3af2bf16f19148a07051dac1#egg=DCNv2', '--user'])
    sys.path.insert(0, "./src/dcnv2")
    import dcn_v2

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, 'lib')
add_path(lib_path)
