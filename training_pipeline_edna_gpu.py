
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

# Use argparse to read this from the terminal

# train_root = '/media/HDD_disk/tomas/ICHOIR/fork/Retrieval-based-Voice-Conversion-WebUI'

# param_dict = {
#     'exp_dir': 'maria-60', 
#     'trainset_dir': f'{train_root}/data/1_16k_wavs', 
#     'sr' : "40k",
#     'num_proc': 54,
#     'f0method' : "pm", 
#     'if_f0' : True,
#     'version' : "v2",
#     'gpus_rmvpe' : '0-0',
#     'spk_id' : 0,
#     'save_epoch': 20,
#     'total_epoch': 60,
#     'batch_size': 20,
#     'if_save_latest': 'No',
#     'if_cache_gpu': 'No',
#     'if_save_every_weights': 'No',
#     'pretrained_G': 'assets/pretrained_v2/f0G40k.pth',
#     'pretrained_D': 'assets/pretrained_v2/f0D40k.pth',
#     'gpus': '0'
# }

######################################################################
parser = argparse.ArgumentParser(
                    prog='Training Pipeline',
                    description='Runs training pipeline for RVC',
                    epilog='Ask me for help')
parser.add_argument('parfile') 

args = parser.parse_args()
json_file = open(args.parfile)
param_dict = json.load(json_file)
print(param_dict)
json_file.close()

######################################################################

logger = logging.getLogger(__name__)

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