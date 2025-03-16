import shutil
import os
import glob

root = '/Users/tomasandrade/Documents/BSC/ICHOIR/organos'
folder_path = f'{root}'

out_dir = f'{folder_path}/flat'
try:
    os.mkdir(out_dir)
except:
    pass

# Construct the search pattern dynamically for .wav files
search_pattern = os.path.join(folder_path, "**", "*.wav")

# Find all .wav files inside the directory and subdirectories
wav_files = glob.glob(search_pattern, recursive=True)

# Print results
for file in wav_files:
    target = file.split(root)[-1][1:].replace('/','_')
    target_path = f'{out_dir}/{target}'
    print(f'{file} --> {target_path}')
    shutil.copy(file, target_path)
