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
    print('------------------  Enter preprocess')
    sr = sr_dict[sr]
    
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

def extract_f0_feature(gpus, n_p, f0method, if_f0, exp_dir, version, gpus_rmvpe,
                       config_vars = None, 
                       now_dir = None,
                       logger = None):
    
    gpus = gpus.split("-")

    print(f'------------------ now dir {now_dir}') 
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    f = open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "w")
    f.close()

    if if_f0:
        if f0method != "rmvpe_gpu":
            cmd = (
                '"%s" infer/modules/train/extract/extract_f0_print.py "%s/logs/%s" %s %s'
                % (
                    config_vars['python_cmd'],
                    now_dir,
                    exp_dir,
                    n_p,
                    f0method,
                )
            )
            logger.info("Execute: " + cmd)
            p = Popen(
                cmd, shell=True, cwd=now_dir
            )  # , stdin=PIPE, stdout=PIPE,stderr=PIPE
            # 煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
            # done = [False]
            # threading.Thread(
            #     target=if_done,
            #     args=(
            #         done,
            #         p,
            #     ),
            # ).start()

    leng = len(gpus)
    ps = []
    for idx, n_g in enumerate(gpus):
        cmd = (
            '"%s" infer/modules/train/extract_feature_print.py %s %s %s %s "%s/logs/%s" %s %s'
            % (
                config_vars['python_cmd'],
                config_vars['device'],
                leng,
                idx,
                n_g,
                now_dir,
                exp_dir,
                version,
                config_vars['is_half']
            )
        )
        logger.info("Execute: " + cmd)
        p = Popen(
            cmd, shell=True, cwd=now_dir
        )  # , shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=now_dir
        ps.append(p)
        p.communicate()


version_config_list = [
    "v1/32k.json",
    "v1/40k.json",
    "v1/48k.json",
    "v2/48k.json",
    "v2/32k.json",
]

def load_config_json() -> dict:
    d = {}
    for config_file in version_config_list:
        p = f"configs/inuse/{config_file}"
        if not os.path.exists(p):
            shutil.copy(f"configs/{config_file}", p)
        with open(f"configs/inuse/{config_file}", "r") as f:
            d[config_file] = json.load(f)
    return d

def if_done(done, p):
    while 1:
        if p.poll() is None:
            sleep(0.5)
        else:
            break
    done[0] = True


def click_train(
    exp_dir1,
    sr2,
    if_f0,
    spk_id,
    save_epoch,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
    config_vars = None, 
    now_dir = None,
    logger = None
):
    print('--------- Inside click_train')

    json_config = load_config_json()

    # 生成filelist
    print(f'------------------ now dir {now_dir}')
    exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    gt_wavs_dir = "%s/0_gt_wavs" % (exp_dir)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    if if_f0:
        f0_dir = "%s/2a_f0" % (exp_dir)
        f0nsf_dir = "%s/2b-f0nsf" % (exp_dir)
        names = (
            set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
            & set([name.split(".")[0] for name in os.listdir(feature_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(feature_dir)]
        )
    opt = []
    for name in names:
        if if_f0:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    f0_dir.replace("\\", "\\\\"),
                    name,
                    f0nsf_dir.replace("\\", "\\\\"),
                    name,
                    spk_id,
                )
            )
        else:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    spk_id,
                )
            )
    fea_dim = 256 if version19 == "v1" else 768
    if if_f0:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, now_dir, now_dir, spk_id)
            )
    else:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, spk_id)
            )
    shuffle(opt)
    with open("%s/filelist.txt" % exp_dir, "w") as f:
        f.write("\n".join(opt))
    logger.debug("Write filelist done")
    
    logger.info("Use gpus: %s", str(gpus16))
    if pretrained_G14 == "":
        logger.info("No pretrained Generator")
    if pretrained_D15 == "":
        logger.info("No pretrained Discriminator")
    if version19 == "v1" or sr2 == "40k":
        config_path = "v1/%s.json" % sr2
    else:
        config_path = "v2/%s.json" % sr2
    config_save_path = os.path.join(exp_dir, "config.json")
    if not pathlib.Path(config_save_path).exists():
        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(
                json_config[config_path],
                f,
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
            )
            f.write("\n")
    if gpus16:
        cmd = (
            '"%s" infer/modules/train/train.py -e "%s" -sr %s -f0 %s -bs %s -g %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'
            % (
                config_vars['python_cmd'], 
                exp_dir1,
                sr2,
                1 if if_f0 else 0,
                batch_size12,
                gpus16,
                total_epoch11,
                save_epoch,
                "-pg %s" % pretrained_G14 if pretrained_G14 != "" else "",
                "-pd %s" % pretrained_D15 if pretrained_D15 != "" else "",
                1 if if_save_latest13 == 'Yes' else 0,
                1 if if_cache_gpu17 == 'Yes' else 0,
                1 if if_save_every_weights18 == 'Yes' else 0,
                version19,
            )
        )
    else:
        cmd = (
            '"%s" infer/modules/train/train.py -e "%s" -sr %s -f0 %s -bs %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'
            % (
                config_vars['python_cmd'],
                exp_dir1,
                sr2,
                1 if if_f0 else 0,
                batch_size12,
                total_epoch11,
                save_epoch,
                "-pg %s" % pretrained_G14 if pretrained_G14 != "" else "",
                "-pd %s" % pretrained_D15 if pretrained_D15 != "" else "",
                1 if if_save_latest13 == 'Yes' else 0,
                1 if if_cache_gpu17 == 'Yes' else 0,
                1 if if_save_every_weights18 == 'Yes' else 0,
                version19,
            )
        )
    logger.info("Execute: " + cmd)
    p = Popen(cmd, shell=True, cwd=now_dir)
    p.wait()
    return "训练结束, 您可查看控制台训练日志或实验文件夹下的train.log"