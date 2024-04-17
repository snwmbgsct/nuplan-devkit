from bokeh.io import output_notebook

from tutorials.utils.tutorial_utils import visualize_nuplan_scenarios


import os
os.environ["BOKEH_ALLOW_WS_ORIGIN"] = "localhost:8888"
NUPLAN_DATA_ROOT = os.getenv('NUPLAN_DATA_ROOT', '/home/arc/nuplan')
NUPLAN_MAPS_ROOT = os.getenv('NUPLAN_MAPS_ROOT', '/home/arc/nuplan/dataset/maps')
NUPLAN_DB_FILES = os.getenv('NUPLAN_DB_FILES', '/home/arc/nuplan/dataset/nuplan-v1.1/mini')
NUPLAN_MAP_VERSION = os.getenv('NUPLAN_MAP_VERSION', 'nuplan-maps-v1.0')
visualize_nuplan_scenarios(
    data_root=NUPLAN_DATA_ROOT,
    db_files=NUPLAN_DB_FILES,
    map_root=NUPLAN_MAPS_ROOT,
    map_version=NUPLAN_MAP_VERSION
)