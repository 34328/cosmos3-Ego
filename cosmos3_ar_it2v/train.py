"""Register pure-video AR recipe, then enter the native Cosmos3 trainer."""
import os
import runpy
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
from . import config
if __name__ == '__main__':
    runpy.run_module('cosmos_framework.scripts.train', run_name='__main__')
