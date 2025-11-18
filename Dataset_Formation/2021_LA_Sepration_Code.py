import os
import shutil

# === User inputs ===
txt_file = r"C:\Users\admin\Downloads\trial_metadata.txt\trial_metadata.txt"    # path to your .txt file
audio_folder = r"F:\ASVspoof2021_LA_eval\ASVspoof2021_LA_eval\flac"   # folder where all .flac files are stored
output_folder = r"F:\2021_eval_LA"     # folder where 'bonafide' and 'spoof' will be created

# === Create output folders ===
bonafide_dir = os.path.join(output_folder, "bonafide")
spoof_dir = os.path.join(output_folder, "spoof")
os.makedirs(bonafide_dir, exist_ok=True)
os.makedirs(spoof_dir, exist_ok=True)

# === Parse the text file ===
with open(txt_file, "r") as f:
    lines = f.readlines()

for line in lines:
    parts = line.strip().split()
    if len(parts) >= 5:
        filename = parts[1]  # e.g., LA_T_9557645
        label = parts[-3]    # 'bonafide' or 'spoof'
        flac_name = f"{filename}.flac"
        src_path = os.path.join(audio_folder, flac_name)
        
        # Decide where to move/copy
        if os.path.exists(src_path):
            if label.lower() == "bonafide":
                shutil.copy(src_path, os.path.join(bonafide_dir, flac_name))
            elif label.lower() == "spoof":
                shutil.copy(src_path, os.path.join(spoof_dir, flac_name))
        else:
            print(f"File not found: {flac_name}")

print("✅ Files divided successfully!")
