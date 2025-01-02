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

# config = Config()

# Taken from Config defaults
python_cmd = os.environ.get("python_cmd")
preprocess_per = float(os.environ.get("preprocess_per"))
noparallel = bool(os.environ.get("noparallel"))
is_half = bool(os.environ.get("is_half"))
device = os.environ.get("device")
n_cpu = int(os.environ.get("n_cpu"))

for var in [python_cmd, preprocess_per, noparallel, is_half, device]:
    print(f'The value is {var}')
    print(f'The type of var {type(var)}')

logger = logging.getLogger(__name__)

#############################################################################

# TA MESSAGE from preprocess_dataset --------
# TA MESSAGE config.python_cmd = /home/tomas/.pyenv/versions/rvc/bin/python
# TA MESSAGE config.noparallel = False
# TA MESSAGE config.noparallel = 3.7
# TA MESSAGE sr = 40000
# TA MESSAGE n_p = 54
# TA MESSAGE gpus pre split = 0
# TA MESSAGE from extract_f0_feature --------
# TA MESSAGE config.python_cmd = /home/tomas/.pyenv/versions/rvc/bin/python
# TA MESSAGE config.device = cuda:0
# TA MESSAGE config.is_half = True
# TA MESSAGE n_p = 54
# TA MESSAGE gpus post split = ['0']
# TA MESSAGE if_f0 = True
# TA MESSAGE version19 = v2
# TA MESSAGE gpus_rmvpe = 0-0
# TA MESSAGE from click_train --------
# TA MESSAGE pretrained_G14 = assets/pretrained_v2/f0G40k.pth
# TA MESSAGE pretrained_D15 = assets/pretrained_v2/f0D40k.pth
# TA MESSAGE gpus16 = 0
# TA MESSAGE train_index ------------
# TA MESSAGE config.n_cpu = 80

#root = '/Users/tomasandrade/Documents/BSC/ICHOIR/fork/Retrieval-based-Voice-Conversion-WebUI'
root = '/media/HDD_disk/tomas/ICHOIR/fork/Retrieval-based-Voice-Conversion-WebUI'
#trainset_dir = f'{root}/data/1_16k_wavs'
trainset_dir = 'data/small_dataset'
exp_dir = 'maria-50'
sr = "40k"
num_proc = 54

#"Select the pitch extraction algorithm: when extracting singing, 
# you can use 'pm' to speed up. For high-quality speech with fast 
# performance, but worse CPU usage, you can use 'dio'. 'harvest' 
# results in better quality but is slower.  'rmvpe' has the best 
# results and consumes less CPU/GPU",
choices_f0method8=["pm", "harvest", "dio", "rmvpe", "rmvpe_gpu"]
f0method = "pm"

# # "Enter the GPU index(es) separated by '-', e.g., 
# # 0-1-2 to use GPU 0, 1, and 2:"
# gpus6 = '' 

# "Whether the model has pitch guidance 
# (required for singing, optional for speech):"
if_f0 = True

# "Version"
choices_version =["v1", "v2"]
version = "v2"

# "Enter the GPU index(es) separated by '-', e.g., 
# 0-0-1 to use 2 processes in GPU0 and 1 process in GPU1",
gpus_rmvpe = '0-0' # for no gpus

# speaker id???
spk_id = 0

# Save frequency (5)
save_epoch = 25

#Total training epochs (20)
total_epoch = 50

# Batch size per GPU (1)
batch_size = 40

# Save only the latest '.ckpt' file to save disk space: (No)
if_save_latest = 'No'

# Cache all training sets to GPU memory. Caching small datasets 
# (less than 10 minutes) can speed up training, but caching large datasets 
# will consume a lot of GPU memory and may not provide much speed improvement: (No)
if_cache_gpu = 'No'

# Save a small final model to the 'weights' folder at each save point: (No)
if_save_every_weights = 'No'

# Load pre-trained base model G path: (assets/pretrained_v2/f0G40k.pth)
pretrained_G = 'assets/pretrained_v2/f0G40k.pth'

# Load pre-trained base model D path: (assets/pretrained_v2/f0D40k.pth)
pretrained_D = 'assets/pretrained_v2/f0D40k.pth'

# Enter the GPU index(es) separated by '-', e.g., 0-1-2 to use GPU 0, 1, 
# and 2: (None but -??)
gpus = '0'

#############################################################################

sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}

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

#from infer-web.py
def preprocess_dataset(trainset_dir, exp_dir, sr, n_p):
    sr = sr_dict[sr]
    print('making dirs')
    print(f'------------------ now dir {now_dir}')
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    f = open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "w")
    f.close()

    cmd = '"%s" infer/modules/train/preprocess.py "%s" %s %s "%s/logs/%s" %s %.1f' % (
        python_cmd,
        trainset_dir,
        sr,
        n_p,
        now_dir,
        exp_dir,
        noparallel,
        preprocess_per,
    )
    
    logger.info("Execute: " + cmd)
    p = Popen(cmd, shell=True)

    # threading stuff 
    # 煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
    
    done = [False]
    threading.Thread(
        target=if_done,
        args=(
            done,
            p,
        ),
    ).start()
    while 1:
        with open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "r") as f:
            yield (f.read())
        sleep(1)
        if done[0]:
            break
    with open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "r") as f:
        log = f.read()
    logger.info(log)
    yield log

def extract_f0_feature(gpus, n_p, f0method, if_f0, exp_dir, version, gpus_rmvpe):
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
                    python_cmd,
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
                python_cmd,
                device,
                leng,
                idx,
                n_g,
                now_dir,
                exp_dir,
                version,
                is_half
            )
        )
        logger.info("Execute: " + cmd)
        p = Popen(
            cmd, shell=True, cwd=now_dir
        )  # , shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=now_dir
        ps.append(p)

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
                python_cmd, 
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
                python_cmd,
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

preprocess_dataset(trainset_dir, exp_dir, sr, num_proc)

# Need to give enough time for some folders to be created. This seems
# very buggy! maybe create the folders from the beginning, instead of
# doing it in infer/modules/train/preprocess.py
print('Im only sleeping')
sleep(5)

extract_f0_feature(gpus,
                    num_proc,
                    f0method,
                    if_f0,
                    exp_dir,
                    version,
                    gpus_rmvpe)

print('Im only sleeping')
sleep(5)

click_train(
    exp_dir,
    sr,
    if_f0,
    spk_id,
    save_epoch,
    total_epoch,
    batch_size,
    if_save_latest,
    pretrained_G,
    pretrained_D,
    gpus,
    if_cache_gpu,
    if_save_every_weights,
    version,
)

# print('Im only sleeping')
# sleep(5)

train_index(exp_dir, version)