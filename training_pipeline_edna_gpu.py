# from random import shuffle
# import pathlib
# import json
# import shutil

# import numpy as np
# from sklearn.cluster import MiniBatchKMeans
# import traceback
# import faiss
# import torch, platform

# from time import sleep
# from subprocess import Popen
# import threading


import os
import sys
import logging

from dotenv import load_dotenv
now_dir = os.getcwd()
sys.path.append(now_dir)
load_dotenv()

import infer.modules.train.training_pipeline as pipeline 

# def read_config_vars():

#     python_cmd = os.environ.get("python_cmd")
#     preprocess_per = float(os.environ.get("preprocess_per"))
#     noparallel = bool(os.environ.get("noparallel"))
#     is_half = bool(os.environ.get("is_half"))
#     device = os.environ.get("device")
#     n_cpu = int(os.environ.get("n_cpu"))

#     config_vars = {'python_cmd': python_cmd, 
#                'preprocess_per': preprocess_per,
#                'noparallel': noparallel,
#                'is_half': is_half,
#                'device': device,
#                'n_cpu': n_cpu}
    
#     return config_vars


train_root = '/media/HDD_disk/tomas/ICHOIR/fork/Retrieval-based-Voice-Conversion-WebUI'

param_dict = {
    'exp_dir': 'small-data_v1', 
    'trainset_dir': f'{train_root}/data/small_dataset',
    'sr' : "40k",
    'num_proc': 54,
    'f0method' : "pm", 
    'if_f0' : True,
    'version' : "v2",
    'gpus_rmvpe' : '0-0',
    'spk_id' : 0,
    'save_epoch': 2,
    'total_epoch': 2,
    'batch_size': 40,
    'if_save_latest': 'No',
    'if_cache_gpu': 'No',
    'if_save_every_weights': 'No',
    'pretrained_G': 'assets/pretrained_v2/f0G40k.pth',
    'pretrained_D': 'assets/pretrained_v2/f0D40k.pth',
    'gpus': '0'
}

logger = logging.getLogger(__name__)




#############################################################################

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