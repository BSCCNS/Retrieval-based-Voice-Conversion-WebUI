from pathlib import Path
import os

from dotenv import load_dotenv
from scipy.io import wavfile

from infer_script.vc_script.modules import VC

load_dotenv()

#######################################################
## model
#######################################################

root = '/media/HDD_disk/tomas/ICHOIR/fork/Retrieval-based-Voice-Conversion-WebUI'

# root_model = f'{root}/assets'
# root_output = 'audio_rvc_output'

# model_name = 'maria-200_rmvpe_gpu'

# # model_pth = f'{root_model}/{model_name}/{model_name}.pth'
# model_pth = f'{root}/assets/weights/maria-200-rmvpe_gpu.pth' 
# index_file = f'{root_model}/{model_name}/added_IVF3808_Flat_nprobe_1_maria-200-rmvpe_gpu_v2.index'

# hubert_path = f'{root_model}/hubert/hubert_base.pt'

#######################################################
## input
#######################################################

#root_input = 'data/ame/'
#input_audio =  f'{root_input}/ame_campana_1.wav'

rvc_dict = {
    "model_name": "maria-200-rmvpe_gpu",
    "index_file": "added_IVF3808_Flat_nprobe_1_maria-200-rmvpe_gpu_v2.index",
    "hubert_file": "hubert_base.pt",
    "input_path": "data/ame/ame_campana_1.wav",
    "f0_method": "rmvpe",
    "protect": 0.33,
    "f0_up_key": 0}

model_name = rvc_dict["model_name"]
index_file = rvc_dict["index_file"]

model_path = f'{root}/assets/{model_name}/{model_name}.pth'
#model_path = f'{model_name}.pth' 
index_path = f'{root}/assets/{model_name}/{index_file}'
hubert_path = f'{root}/assets/hubert/{rvc_dict["hubert_file"]}'
input_path = rvc_dict["input_path"]

wav_name = input_path.split('/')[-1].split('.wav')[-2]
f0_method = rvc_dict["f0_method"]
protect = rvc_dict["protect"]
f0_up_key = rvc_dict["f0_up_key"]

output_path = f'{root}/audio_rvc_output/{wav_name}_by_{model_name}_f0_method_{f0_method}_protect_{protect}_f0_up_key_{f0_up_key}.wav'

vc = VC()
vc.get_vc(model_path, index_file = index_path)


print('--------------- f0 method rmvpe')
tgt_sr, audio_opt, times, _ = vc.vc_inference(1, input_path, #Path(input_audio),
                                                hubert_path = hubert_path,
                                                f0_method = f0_method,
                                                f0_up_key = f0_up_key,
                                                protect = protect)

wavfile.write(output_path, tgt_sr, audio_opt)
