import os
import soundfile as sf
import numpy as np

# Input and output directories
input_folder = '/media/HDD_disk/tomas/ICHOIR/fork/Retrieval-based-Voice-Conversion-WebUI/audio_rvc_output/AMA_INPUT_SMALL_by_violeta_dataset_3albums_titan_40_batch'
output_folder = '/media/HDD_disk/tomas/ICHOIR/fork/Retrieval-based-Voice-Conversion-WebUI/audio_rvc_output/AMA_INPUT_SMALL_by_violeta_dataset_3albums_titan_40_batch_stereo'

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Loop through matching L-channel files
for filename in os.listdir(input_folder):
    if filename.endswith('L_2nd_generation.wav'):
        base_name = filename[:-len('L_2nd_generation.wav')]

        left_path = os.path.join(input_folder, filename)
        right_filename = base_name + 'R_2nd_generation.wav'
        right_path = os.path.join(input_folder, right_filename)

        if not os.path.exists(right_path):
            print(f"Missing right channel for: {base_name}")
            continue

        # Load audio
        left_data, sr_left = sf.read(left_path)
        right_data, sr_right = sf.read(right_path)

        if sr_left != sr_right:
            print(f"Sample rate mismatch for: {base_name}")
            continue

        # Trim to shortest channel length if needed
        min_len = min(len(left_data), len(right_data))
        if len(left_data) != len(right_data):
            print(f"Trimming to match length for: {base_name}")
            left_data = left_data[:min_len]
            right_data = right_data[:min_len]

        # Combine into stereo
        stereo_data = np.column_stack((left_data, right_data))

        # Output file path
        output_path = os.path.join(output_folder, base_name + 'stereo.wav')

        # Write stereo file with forced 24-bit PCM encoding
        sf.write(output_path, stereo_data, sr_left, format='WAV', subtype='PCM_24')

print("Stereo files created successfully with PCM_24 encoding.")
