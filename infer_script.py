from pathlib import Path
import os

from dotenv import load_dotenv
from scipy.io import wavfile

from infer_script.vc_script.modules import VC

load_dotenv()

root = '/Users/tomasandrade/Documents/BSC/ICHOIR/Retrieval-based-Voice-Conversion/'

root_model = f'{root}/assets'
root_output = '.'

model_name = 'maria-200_rmvpe_gpu'

model_pth = f'{root_model}/{model_name}/{model_name}.pth'
index_file = f'{root_model}/{model_name}/added_IVF3808_Flat_nprobe_1_maria-200-rmvpe_gpu_v2.index'

hubert_path = f'{root_model}/hubert/hubert_base.pt'



root_input = '/Users/tomasandrade/Documents/BSC/ICHOIR/Retrieval-based-Voice-Conversion/output'
input_audio =  f'{root_input}/ame_campana_1.wav'

vc = VC()
vc.get_vc(model_pth, index_file = index_file)


tgt_sr, audio_opt, times, _ = vc.vc_inference(1, Path(input_audio), 
                                                hubert_path = hubert_path,
                                                f0_method = "pm",
                                                f0_up_key = 0,
                                                protect = 0.33) 

output_path = f'{root_output}/ame_campana_1_by_{model_name}.wav'
wavfile.write(output_path, tgt_sr, audio_opt)