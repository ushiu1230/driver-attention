import os
import subprocess

root_dir = "./drivergaze360_subset"
print(f"Searching for video files in {root_dir}...")

target_pairs = [("rgb.mp4", "rgb"), ("saliency.mp4", "saliency")]
processed_count = 0

# Set the extraction FPS (1 FPS as recommended in optimization step 3)
TARGET_FPS = 1

for root, dirs, files in os.walk(root_dir):
    # Iterate through the files in the current directory
    for file_name in files:
        for video_name, out_folder_name in target_pairs:
            # Match case-insensitively to avoid filesystem path issues
            if file_name.lower() == video_name.lower():
                video_path = os.path.join(root, file_name)
                output_dir = os.path.join(root, out_folder_name)

                # Skip if the destination folder already contains extracted frames
                if os.path.exists(output_dir) and len([f for f in os.listdir(output_dir) if f.lower().endswith('.jpg')]) > 0:
                    print(f"Already extracted: {output_dir}")
                    continue

                print(f"Extracting: {video_path} -> {output_dir} (Sampling rate: {TARGET_FPS} FPS)")
                os.makedirs(output_dir, exist_ok=True)

                # Extract at a low frame rate (-r 1) to reduce the dataset's storage size
                cmd = f"ffmpeg -y -i \"{video_path}\" -r {TARGET_FPS} -q:v 1 -qmin 1 -start_number 1 \"{output_dir}/%06d.jpg\""
                res = subprocess.run(cmd, shell=True, capture_output=True, text=True)

                if res.returncode == 0:
                    extracted_count = len([f for f in os.listdir(output_dir) if f.lower().endswith('.jpg')])
                    print(f"Successfully extracted {extracted_count} frames.")
                    processed_count += 1
                else:
                    print(f"Error extracting {video_path}: {res.stderr}")

print(f"\nAll done! Processed {processed_count} video files with downsampled rate.")