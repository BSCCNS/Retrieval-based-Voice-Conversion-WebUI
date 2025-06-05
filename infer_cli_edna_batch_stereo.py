from pathlib import Path
import os
import json
import argparse
import glob

from dotenv import load_dotenv
from scipy.io import wavfile

from my_tools.combine_left_right import merge_LR_channels
from my_tools.reshape_folder_structure import flatten_files

from infer_script.vc_script.modules import VC

load_dotenv()

parser = argparse.ArgumentParser(
                    prog='Conversion Pipeline',
                    description='Runs conversion pipeline for RVC',
                    epilog='Ask me for help')
parser.add_argument('parfile') 

def read_param_dict(parser):
    args = parser.parse_args()
    json_file = open(args.parfile)
    param_dict = json.load(json_file)
    json_file.close()

    return param_dict

rvc_dict = read_param_dict(parser)

root = '/media/HDD_disk/tomas/ICHOIR/fork/Retrieval-based-Voice-Conversion-WebUI'

model_name = rvc_dict["model_name"]
index_file = rvc_dict["index_file"]

model_path = f'{model_name}.pth' 
index_path = f'{root}/logs/{model_name}/{index_file}'
hubert_path = f'{root}/assets/hubert/{rvc_dict["hubert_file"]}'
#input_path = rvc_dict["input_path"]
input_folder = rvc_dict["input_folder"]
folder_name = input_folder.split('/')[-1]#.split('.wav')[-2]

files = glob.glob(f'{input_folder}/*.wav')

# conversion parameters
f0_method = rvc_dict["f0_method"]
protect = rvc_dict["protect"]
f0_up_key = rvc_dict["f0_up_key"]
rms_mix_rate = rvc_dict["rms_mix_rate"]
resample_sr = rvc_dict.get("resample_sr", 0)

# output parameters
#experiment_name = f'{wav_name}_by_{model_name}_f0_method_{f0_method}_protect_{protect}_f0_up_key_{f0_up_key}_loop'
experiment_name = f'{folder_name}_by_{model_name}_f0_method_{f0_method}_protect_{protect}_f0_up_key_{f0_up_key}_batch'
experiment_dir = f'{root}/audio_rvc_output/{experiment_name}'

os.mkdir(experiment_dir)

with open(f"{experiment_dir}/metadata.json", "w") as outfile: 
    json.dump(rvc_dict, outfile)

vc = VC()
vc.get_vc(model_path, index_file = index_path)

for input_path in files:
    print(f'--------------- file {input_path}')

    out_wav_name = input_path.split('/')[-1].split('.wav')[-2]
    output_path = f'{experiment_dir}/{out_wav_name}_2nd_generation.wav'
    
    tgt_sr, audio_opt, times, _ = vc.vc_inference(1, input_path, #Path(input_audio),
                                                    hubert_path = hubert_path,
                                                    f0_method = f0_method,
                                                    f0_up_key = f0_up_key,
                                                    protect = protect,
                                                    rms_mix_rate = rms_mix_rate,
                                                    resample_sr = resample_sr)
    
    print(f'--------------- Saving audio to {output_path}')
    wavfile.write(output_path, tgt_sr, audio_opt)

out_folder = merge_LR_channels(experiment_dir)

_ = flatten_files(out_folder)