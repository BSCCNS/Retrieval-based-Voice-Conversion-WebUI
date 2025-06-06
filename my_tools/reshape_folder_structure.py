import os
import shutil
import glob

# Folder containing flattened files
#input_folder = '/media/HDD_disk/tomas/ICHOIR/fork/Retrieval-based-Voice-Conversion-WebUI/audio_rvc_output/AMA_INPUT_SMALL_by_violeta_dataset_3albums_titan_40_batch_stereo'
# Folder to restore subfolder structure
#output_folder = '/media/HDD_disk/tomas/ICHOIR/fork/Retrieval-based-Voice-Conversion-WebUI/audio_rvc_output/AMA_INPUT_SMALL_by_violeta_dataset_3albums_titan_40_batch_stereo_struct'

def to_subfolders(input_folder, output_folder = None):

    if output_folder is None:
        output_folder = f'{input_folder}_struct'

    print('FLATTEN')
    print(f'Input folder: {input_folder}')
    print(f'Output folder: {output_folder}')

    # Create output root if needed
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all files in the flat folder
    for filename in os.listdir(input_folder):
        print(f'------ working on {filename}')
        # Only process files with double underscore separator
        if '__' in filename:
            # Reconstruct relative path
            relative_path = filename.replace('__', os.sep)

            # Full source and destination paths
            src_path = os.path.join(input_folder, filename)
            dst_path = os.path.join(output_folder, relative_path)

            # Create destination directories if needed
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)

            print(f'-------------- Writing file from {src_path} to {dst_path}')
            # Move or copy file (use shutil.move to move instead of copy)
            shutil.copy2(src_path, dst_path)

    print("Flattened files restored to original folder structure.")
    return output_folder


def flatten_folder(source_folder, output_folder = None):

    if output_folder is None:
        output_folder = f'{source_folder}_flat'

    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get all .wav files recursively
    wav_files = glob.glob(os.path.join(source_folder, '**', '*.wav'), recursive=True)

    for file_path in wav_files:
        # Get relative path to source folder
        rel_path = os.path.relpath(file_path, start=source_folder)

        # Replace path separators with double underscores
        flat_filename = rel_path.replace(os.sep, '__')

        # Full destination path
        dest_path = os.path.join(output_folder, flat_filename)

        # Copy file to flattened structure
        shutil.copy2(file_path, dest_path)

    print("Files flattened and copied successfully using '__' as separator.")

if __name__ == '__main__':
    import sys
    arguments = sys.argv[1:]

    source_folder = arguments[0]
    output_folder = arguments[1]

    flatten_folder(source_folder, output_folder = output_folder)

