import glob
import os
import pandas as pd
from pydub import AudioSegment
import wave


root = '/Users/tomasandrade/Documents/BSC/ICHOIR/organos/flat'

files = glob.glob(f'{root}/*.wav', recursive=False)

ls_sr = []
for file in files:

    # Load the WAV file
    audio = AudioSegment.from_wav(file)

    # Get the sample rate
    sample_rate = audio.frame_rate
    ls_sr.append(sample_rate)
    
df = pd.DataFrame(data = ls_sr, columns=['sr'])

print(df['sr'].unique())

min_sr = df['sr'].min()
print(f'Minimum sample rate = {min_sr}')

out_dir = f'{root}/48k'
try:
    os.mkdir(out_dir)
except:
    pass

for file in files[0:3]:

    print(file)
    target = file.split(root)[-1].replace('.wav', '_48k.wav')
    output_file = f'{out_dir}/{target}'

    audio = AudioSegment.from_wav(file)

    if audio.frame_rate >= min_sr:
        downsampled_audio = audio.set_frame_rate(min_sr)

        # Export the modified audio
        downsampled_audio.export(output_file, format="wav")
    
    else:
        print(f'Problem with file {file}! sample rate lower than minimum')


