
import os
import soundfile as sf

def split_LR(input_folder, output_folder = None):

    if output_folder is None:
        output_folder = f'{input_folder}_LR'

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all .wav files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.wav'):
            filepath = os.path.join(input_folder, filename)

            # Get audio metadata
            info = sf.info(filepath)
            print(f'Info for {filepath} \n{info}')

            # Read audio
            data, samplerate = sf.read(filepath)

            # Check if stereo
            if data.ndim != 2 or data.shape[1] != 2:
                print(f"Skipping non-stereo file: {filename}")
                continue

            # Split channels
            left = data[:, 0]
            right = data[:, 1]

            # Output filenames
            base_name = os.path.splitext(filename)[0]
            left_path = os.path.join(output_folder, f"{base_name}_L.wav")
            right_path = os.path.join(output_folder, f"{base_name}_R.wav")

            # Write files, preserving format and subtype
            sf.write(left_path, left, samplerate, format=info.format, subtype=info.subtype)
            sf.write(right_path, right, samplerate, format=info.format, subtype=info.subtype)

    print("All stereo files split and saved with original encoding preserved.")
    return output_folder