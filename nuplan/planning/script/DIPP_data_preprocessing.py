import logging
import os
from typing import Optional

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from nuplan.planning.script.builders.folder_builder import build_training_experiment_folder
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.script.builders.utils.utils_config import update_config_for_training
from nuplan.planning.script.builders.worker_pool_builder import build_worker
from nuplan.planning.script.utils import set_default_path
from nuplan.planning.training.experiments.caching import cache_data
from nuplan.planning.training.experiments.training import TrainingEngine, build_training_engine

# from nuplan.planning.training.modeling.models.DIPP_model import DIPP_train


import os
from pathlib import Path
import tempfile

import hydra






logging.getLogger('numba').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# If set, use the env. variable to overwrite the default dataset and experiment paths
set_default_path()

# If set, use the env. variable to overwrite the Hydra config
CONFIG_PATH = os.getenv('NUPLAN_HYDRA_CONFIG_PATH', 'config/training')

if os.environ.get('NUPLAN_HYDRA_CONFIG_PATH') is not None:
    CONFIG_PATH = os.path.join('../../../../', CONFIG_PATH)

if os.path.basename(CONFIG_PATH) != 'training':
    CONFIG_PATH = os.path.join(CONFIG_PATH, 'training')
CONFIG_NAME = 'default_training'


# @hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> Optional[TrainingEngine]:
    """
    Main entrypoint for training/validation experiments.
    :param cfg: omegaconf dictionary
    """
    # Fix random seed
    pl.seed_everything(cfg.seed, workers=True)

    # Configure logger
    build_logger(cfg)

    # Override configs based on setup, and print config
    update_config_for_training(cfg)

    # Create output storage folder
    build_training_experiment_folder(cfg=cfg)

    # Build worker
    worker = build_worker(cfg)

    if cfg.py_func == 'train':
        # Build training engine
        engine = build_training_engine(cfg, worker)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!",engine.datamodule)
        # Run training
        logger.info('Starting training...')
        engine.trainer.fit(model=engine.model, datamodule=engine.datamodule)
        
        return engine
    elif cfg.py_func == 'test':
        # Build training engine
        engine = build_training_engine(cfg, worker)

        # Test model
        logger.info('Starting testing...')
        engine.trainer.test(model=engine.model, datamodule=engine.datamodule)
        return engine
    elif cfg.py_func == 'cache':
        # Precompute and cache all features
        cache_data(cfg=cfg, worker=worker)
        return None
    elif cfg.py_func == 'DIPP':
        engine = build_training_engine(cfg, worker)
        # Data preprocessing and Training for DIPP
        DIPP_train(datamodule=engine.datamodule)
        return None
    else:
        raise NameError(f'Function {cfg.py_func} does not exist')


if __name__ == '__main__':
    # Location of path with all training configs
    CONFIG_PATH = './config/training'
    CONFIG_NAME = 'default_training'

    # Create a temporary directory to store the cache and experiment artifacts
    SAVE_DIR = Path(tempfile.gettempdir()) / 'tutorial_nuplan_framework'  # optionally replace with persistent dir #NOTES /tmp/tutorial_nuplan_framework
    EXPERIMENT = 'training_vector_experiment'#'training_raster_experiment'
    LOG_DIR = str(SAVE_DIR / EXPERIMENT)

    # Initialize configuration management system
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(config_path=CONFIG_PATH)

    # Compose the configuration
    cfg = hydra.compose(config_name=CONFIG_NAME, overrides=[
        f'group={str(SAVE_DIR)}', #NOTES: default_experiment 
        f'cache.cache_path={str(SAVE_DIR)}/cache', #NOTES: default_training
        f'experiment_name={EXPERIMENT}', #NOTES: default_experiment
        'py_func=train', #NOTES: default_training
        '+training=training_DIPP_model',  # raster model that consumes ego, agents and map raster layers and regresses the ego's trajectory #NOTES:nuplan/planning/script/config/common/model  [model in nuplan/planning/script/config/common/default_common.yaml]
        'scenario_builder=nuplan_mini',  # use nuplan mini database #NOTES:default_common
        'scenario_filter.limit_total_scenarios=500',  # Choose 500 scenarios to train with #NOTES:default_common
        'lightning.trainer.params.accelerator=ddp_spawn',  # ddp is not allowed in interactive environment, using ddp_spawn instead - this can bottleneck the data pipeline, it is recommended to run training outside the notebook #NOTES:nuplan/planning/script/config/training/lightning/default_lightning.yaml
        'lightning.trainer.params.max_epochs=10',
        'data_loader.params.batch_size=8', #NOTES:nuplan/planning/script/config/training/data_loader/default_data_loader.yaml
        'data_loader.params.num_workers=8',#NOTES: [in nuplan/planning/script/config/training/data_loader/default_data_loader.yaml]
        # 'splitter=nuplan'
    ])
    
    main(cfg)
