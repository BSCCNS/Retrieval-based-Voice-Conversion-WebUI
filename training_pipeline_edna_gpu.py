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



def train_index(exp_dir1, version19):
    # exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)

    print('------------------ Inside train_index')
    outside_index_root = os.getenv("outside_index_root")
    # n_cpu = 8
    print(f'------------------ now dir {now_dir}')
    exp_dir = "logs/%s" % (exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    if not os.path.exists(feature_dir):
        return "请先进行特征提取!"
    listdir_res = list(os.listdir(feature_dir))
    if len(listdir_res) == 0:
        return "请先进行特征提取！"
    infos = []
    npys = []
    for name in sorted(listdir_res):
        phone = np.load("%s/%s" % (feature_dir, name))
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    
    if big_npy.shape[0] > 2e5:
        infos.append("Trying doing kmeans %s shape to 10k centers." % big_npy.shape[0])
        #yield "\n".join(infos)
        try:
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=True,
                    #batch_size=256 * config.n_cpu,
                    batch_size=256 * n_cpu,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
        except:
            info = traceback.format_exc()
            logger.info(info)
            infos.append(info)
            #yield "\n".join(infos)

    np.save("%s/total_fea.npy" % exp_dir, big_npy)
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    infos.append("%s,%s" % (big_npy.shape, n_ivf))
    #yield "\n".join(infos)
    
    index = faiss.index_factory(256 if version19 == "v1" else 768, "IVF%s,Flat" % n_ivf)
    # index = faiss.index_factory(256if version19=="v1"else 768, "IVF%s,PQ128x4fs,RFlat"%n_ivf)
    infos.append("training")
    
    #yield "\n".join(infos)
    
    index_ivf = faiss.extract_index_ivf(index)  #
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(
        index,
        "%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )
    infos.append("adding")
    #yield "\n".join(infos)
    
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])
    faiss.write_index(
        index,
        "%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )
    infos.append(
        "成功构建索引 added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (n_ivf, index_ivf.nprobe, exp_dir1, version19)
    )
    try:
        link = os.link if platform.system() == "Windows" else os.symlink
        link(
            "%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index"
            % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
            "%s/%s_IVF%s_Flat_nprobe_%s_%s_%s.index"
            % (
                outside_index_root,
                exp_dir1,
                n_ivf,
                index_ivf.nprobe,
                exp_dir1,
                version19,
            ),
        )
        infos.append("链接索引到外部-%s" % (outside_index_root))
    except:
        infos.append("链接索引到外部-%s失败" % (outside_index_root))

    print('Training index finished!')
    # faiss.write_index(index, '%s/added_IVF%s_Flat_FastScan_%s.index'%(exp_dir,n_ivf,version19))
    # infos.append("成功构建索引，added_IVF%s_Flat_FastScan_%s.index"%(n_ivf,version19))
    #yield "\n".join(infos)

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

    

# train_index(exp_dir, version)