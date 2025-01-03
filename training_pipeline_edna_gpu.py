'''
use : python training_pipeline_edna_gpu.py <parfile>.json
'''

import os
import sys
#import json
import logging
import argparse

from dotenv import load_dotenv
import infer.modules.train.training_pipeline as pipeline 

now_dir = os.getcwd()
sys.path.append(now_dir)
load_dotenv()

parser = argparse.ArgumentParser(
                    prog='Training Pipeline',
                    description='Runs training pipeline for RVC',
                    epilog='Ask me for help')
parser.add_argument('parfile') 


logger = logging.getLogger(__name__)

param_dict = pipeline.read_param_dict(parser)
config_vars = pipeline.read_config_vars()

pipeline.preprocess_dataset(param_dict,
                            config_vars = config_vars, 
                            now_dir = now_dir,
                            logger = logger)

pipeline.extract_f0_feature(param_dict,
                            config_vars = config_vars, 
                            now_dir = now_dir,
                            logger = logger)

pipeline.click_train(param_dict,
                    config_vars = config_vars, 
                    now_dir = now_dir,
                    logger = logger)

pipeline.train_index(param_dict,
                    config_vars = config_vars, 
                    now_dir = now_dir,
                    logger = logger)