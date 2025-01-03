
import os
import sys
import json
import logging
import argparse

from dotenv import load_dotenv
now_dir = os.getcwd()
sys.path.append(now_dir)
load_dotenv()

import infer.modules.train.training_pipeline as pipeline 

######################################################################
parser = argparse.ArgumentParser(
                    prog='Training Pipeline',
                    description='Runs training pipeline for RVC',
                    epilog='Ask me for help')
parser.add_argument('parfile') 

def read_param_dict(parser):
    args = parser.parse_args()
    json_file = open(args.parfile)
    param_dict = json.load(json_file)
    json_file.close()

    return param_dict

######################################################################

logger = logging.getLogger(__name__)

param_dict = read_param_dict(parser)
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