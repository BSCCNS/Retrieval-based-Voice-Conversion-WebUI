from pydub import AudioSegment
from pydub.utils import make_chunks
import glob
import os

out_dir = '/Users/tomasandrade/Desktop/violeta/wav_mono_48k_spliced'
os.mkdir(out_dir)

chunk_length_ms = 1000*8 # pydub calculates in millisec

files = glob.glob('/Users/tomasandrade/Desktop/violeta/wav_mono_48k/*.wav', recursive=False)
print(files)


for file in files:

    print(f'------- Working on file {file}')

    myaudio = AudioSegment.from_file(file , "wav") 
    chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec
    name = file.split('/')[-1].split('.wav')[-2].replace(' ', '_')

    print(name)

    for i, chunk in enumerate(chunks):
        chunk_name = f"{name}_{i}.wav"
        print(f"exporting {chunk_name}")
        chunk.export(f'{out_dir}/{chunk_name}', format="wav")