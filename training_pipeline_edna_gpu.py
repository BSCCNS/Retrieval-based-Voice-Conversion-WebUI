import os
import sys
from dotenv import load_dotenv
from random import shuffle
import pathlib
import json
import shutil

import numpy as np
from sklearn.cluster import MiniBatchKMeans
import traceback
import faiss
import torch, platform

now_dir = os.getcwd()
sys.path.append(now_dir)
load_dotenv()

from time import sleep
from subprocess import Popen
import threading
import logging

import infer.modules.train.training_pipeline as pipeline #import preprocess_dataset

# config = Config()

# Taken from Config defaults

def read_config_vars():

    python_cmd = os.environ.get("python_cmd")
    preprocess_per = float(os.environ.get("preprocess_per"))
    noparallel = bool(os.environ.get("noparallel"))
    is_half = bool(os.environ.get("is_half"))
    device = os.environ.get("device")
    n_cpu = int(os.environ.get("n_cpu"))

    config_vars = {'python_cmd': python_cmd, 
               'preprocess_per': preprocess_per,
               'noparallel': noparallel,
               'is_half': is_half,
               'device': device,
               'n_cpu': n_cpu}
    
    return config_vars


config_vars = read_config_vars()

root = '/media/HDD_disk/tomas/ICHOIR/fork/Retrieval-based-Voice-Conversion-WebUI'

param_dict = {
    'exp_dir': 'maria-2', 
    'trainset_dir': f'{root}/data/1_16k_wavs',
    'sr' : "40k",
    'num_proc': 54,
    'f0method' : "pm", # ["pm", "harvest", "dio", "rmvpe", "rmvpe_gpu"]
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

# for var in [python_cmd, preprocess_per, noparallel, is_half, device]:
#     print(f'The value is {var}')
#     print(f'The type of var {type(var)}')

logger = logging.getLogger(__name__)

#############################################################################

#root = '/Users/tomasandrade/Documents/BSC/ICHOIR/fork/Retrieval-based-Voice-Conversion-WebUI'
# root = '/media/HDD_disk/tomas/ICHOIR/fork/Retrieval-based-Voice-Conversion-WebUI'
# trainset_dir = f'{root}/data/1_16k_wavs'
# #trainset_dir = 'data/small_dataset'
# exp_dir = 'maria-100'
# sr = "40k"
# num_proc = 54

#"Select the pitch extraction algorithm: when extracting singing, 
# you can use 'pm' to speed up. For high-quality speech with fast 
# performance, but worse CPU usage, you can use 'dio'. 'harvest' 
# results in better quality but is slower.  'rmvpe' has the best 
# results and consumes less CPU/GPU",
# choices_f0method8=["pm", "harvest", "dio", "rmvpe", "rmvpe_gpu"]
# f0method = "pm"

# # "Enter the GPU index(es) separated by '-', e.g., 
# # 0-1-2 to use GPU 0, 1, and 2:"
# gpus6 = '' 

# "Whether the model has pitch guidance 
# (required for singing, optional for speech):"
# if_f0 = True

# "Version"
# choices_version =["v1", "v2"]
# version = "v2"

# "Enter the GPU index(es) separated by '-', e.g., 
# 0-0-1 to use 2 processes in GPU0 and 1 process in GPU1",
#gpus_rmvpe = '0-0' # for no gpus

# speaker id???
#spk_id = 0

# Save frequency (5)
#save_epoch = 25

#Total training epochs (20)
#total_epoch = 100

# Batch size per GPU (1)
#batch_size = 20

# Save only the latest '.ckpt' file to save disk space: (No)
#if_save_latest = 'No'

# Cache all training sets to GPU memory. Caching small datasets 
# (less than 10 minutes) can speed up training, but caching large datasets 
# will consume a lot of GPU memory and may not provide much speed improvement: (No)
#if_cache_gpu = 'No'

# Save a small final model to the 'weights' folder at each save point: (No)
#if_save_every_weights = 'No'

# Load pre-trained base model G path: (assets/pretrained_v2/f0G40k.pth)
#pretrained_G = 'assets/pretrained_v2/f0G40k.pth'

# Load pre-trained base model D path: (assets/pretrained_v2/f0D40k.pth)
#pretrained_D = 'assets/pretrained_v2/f0D40k.pth'

# Enter the GPU index(es) separated by '-', e.g., 0-1-2 to use GPU 0, 1, 
# and 2: (None but -??)
#gpus = '0'

#############################################################################

pipeline.preprocess_dataset(param_dict['trainset_dir'], 
                    param_dict['exp_dir'], 
                    param_dict['sr'], 
                    param_dict['num_proc'], 
                    config_vars = config_vars, 
                    now_dir = now_dir,
                    logger = logger)


pipeline.extract_f0_feature(param_dict['gpus'],
                    param_dict['num_proc'],
                    param_dict['f0method'],
                    param_dict['if_f0'],
                    param_dict['exp_dir'],
                    param_dict['version'],
                    param_dict['gpus_rmvpe'],
                    config_vars = config_vars, 
                    now_dir = now_dir,
                    logger = logger)

pipeline.click_train(
    param_dict['exp_dir'],
    param_dict['sr'],
    param_dict['if_f0'],
    param_dict['spk_id'],
    param_dict['save_epoch'],
    param_dict['total_epoch'],
    param_dict['batch_size'],
    param_dict['if_save_latest'],
    param_dict['pretrained_G'],
    param_dict['pretrained_D'],
    param_dict['gpus'],
    param_dict['if_cache_gpu'],
    param_dict['if_save_every_weights'],
    param_dict['version'],
    config_vars = config_vars, 
    now_dir = now_dir,
    logger = logger
)

pipeline.train_index(param_dict['exp_dir'], 
                    param_dict['version'],
                    config_vars = config_vars, 
                    now_dir = now_dir,
                    logger = logger)