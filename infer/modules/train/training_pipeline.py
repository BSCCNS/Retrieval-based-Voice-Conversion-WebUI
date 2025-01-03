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

sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}


def preprocess_dataset(trainset_dir, exp_dir, sr, n_p, 
                       config_vars = None, 
                       now_dir = None,
                       logger = None):
    sr = sr_dict[sr]
    print('------------------  Enter preprocess')
    
    print(f'now_dir: {now_dir}')

    print('Making dirs')

    #real_exp_dir = "%s/logs/%s" % (now_dir, exp_dir)

    real_exp_dir = f"{now_dir}/logs/{exp_dir}"
    gt_wavs_dir =  f"{now_dir}/logs/{exp_dir}/0_gt_wavs" 
    wavs16k_dir =  f"{now_dir}/logs/{exp_dir}/1_16k_wavs"

    print(f'Creating real_exp_dir dir: {real_exp_dir}')
    print(f'Creating gt_wavs_dir dir: {gt_wavs_dir}')
    print(f'Creating wavs16k_dir dir: {wavs16k_dir}')

    #os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)   
    os.makedirs(real_exp_dir, exist_ok=True) 
    os.makedirs(gt_wavs_dir, exist_ok=True)
    os.makedirs(wavs16k_dir, exist_ok=True)

    f = open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "w")
    f.close()

    cmd = '"%s" infer/modules/train/preprocess.py "%s" %s %s "%s/logs/%s" %s %.1f' % (
        config_vars['python_cmd'],
        trainset_dir,
        sr,
        n_p,
        now_dir,
        exp_dir,
        config_vars['noparallel'],
        config_vars['preprocess_per'],
    )
    
    logger.info("Execute: " + cmd)
    p = Popen(cmd, shell=True)
    p.communicate()

    # threading stuff 
    # 煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
    
    # done = [False]
    # threading.Thread(
    #     target=if_done,
    #     args=(
    #         done,
    #         p,
    #     ),
    # ).start()
    # while 1:
    #     with open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "r") as f:
    #         yield (f.read())
    #     sleep(1)
    #     if done[0]:
    #         break
    # with open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "r") as f:
    #     log = f.read()
    # logger.info(log)
    # yield log
