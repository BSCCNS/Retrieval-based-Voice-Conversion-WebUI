import os
import sys
from dotenv import load_dotenv
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
import logging

from infer.modules.train.training_pipeline import training_pipeline

# config = Config()

# read config

now_dir = os.getcwd()
sys.path.append(now_dir)
load_dotenv()

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

root = '/media/HDD_disk/tomas/ICHOIR/fork/Retrieval-based-Voice-Conversion-WebUI'

param_dict = {
    'exp_dir': 'maria-40', 
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

#############################################################################

# initiate logger
logger = logging.getLogger(__name__)

training_pipeline(param_dict, config_vars = config_vars, 
                            now_dir = now_dir, 
                            logger = logger)

