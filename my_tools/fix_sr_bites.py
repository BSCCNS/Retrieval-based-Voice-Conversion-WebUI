import glob
import os
import pandas as pd
from pydub import AudioSegment
import wave
import argparse

parser = argparse.ArgumentParser(
                    prog='Conversion Pipeline',
                    description='Runs conversion pipeline for RVC',
                    epilog='Ask me for help')
parser.add_argument('root') 

args = parser.parse_args()
root = args.root

#root = '.'#'/Users/tomasandrade/Documents/BSC/ICHOIR/organos/flat'

files = glob.glob(f'{root}/*.wav', recursive=False)

# target_sr = 16000
# target_bites = 0##

TARGET_SAMPLE_RATE = 16000
TARGET_BIT_DEPTH = "pcm_s32le"

out_dir = f'{root}/resample'
try:
    os.mkdir(out_dir)
except:
    pass

for file in files:

    print(file)
    target = file.split(root)[-1] #.replace('.wav', '_48k.wav')
    output_file = f'{out_dir}{target}'

    #print(output_file)

    audio = AudioSegment.from_wav(file)

    downsampled_audio = audio.set_frame_rate(TARGET_SAMPLE_RATE)
    downsampled_audio.export(output_file, format="wav", parameters=["-acodec", TARGET_BIT_DEPTH])
    
    


