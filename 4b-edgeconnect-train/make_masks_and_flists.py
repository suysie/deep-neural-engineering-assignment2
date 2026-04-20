import os
import glob
import cv2
import numpy as np

# Mappen configuratie
DATA_ROOT = "/workspace/data"
OUTPUT_DIR = "./datasets"

# Zorg dat de output map voor de flists bestaat
os.makedirs(OUTPUT_DIR, exist_ok=True)

splits = {
    "train": {
        "target": os.path.join(DATA_ROOT, "train_1_target"),
        "edge": os.path.join(DATA_ROOT, "train_1_skeleton"),
        "input": os.path.join(DATA_ROOT, "train_1_input"),
        "mask": os.path.join(DATA_ROOT, "train_1_mask")
    },
    "val": {
        "target": os.path.join(DATA_ROOT, "val_target"),
        "edge": os.path.join(DATA_ROOT, "val_skeleton"),
        "input": os.path.join(DATA_ROOT, "val_input"),
        "mask": os.path.join(DATA_ROOT, "val_mask")
    },
    "test": {
        "target": os.path.join(DATA_ROOT, "test_target"),
        "edge": os.path.join(DATA_ROOT, "test_skeleton"),
        "input": os.path.join(DATA_ROOT, "test_input"),
        "mask": os.path.join(DATA_ROOT, "test_mask")
    }
}

def generate_masks_and_flist(paths, split_name):
    target_dir = paths["target"]
    input_dir = paths["input"]
    mask_dir = paths["mask"]
    
    if not os.path.exists(target_dir) or not os.path.exists(input_dir):
        print(f"Sla {split_name} over: Input of Target map ontbreekt.")
        return
        
    os.makedirs(mask_dir, exist_ok=True)
    
    # Zoek bestanden
    target_files = sorted(glob.glob(os.path.join(target_dir, "*.png")) + glob.glob(os.path.join(target_dir, "*.jpg")))
    
    mask_paths = []
    
    print(f"Genereer maskers voor {split_name} ({len(target_files)} bestanden)...")
    
    for target_path in target_files:
        filename = os.path.basename(target_path)
        
        # FIX: Verander _target in _input voor het zoeken in de input map
        input_filename = filename.replace("_target", "_input")
        input_path = os.path.join(input_dir, input_filename)
        
        mask_path = os.path.join(mask_dir, filename.replace("_target", "_mask"))
        
        if not os.path.exists(input_path):
            print(f"Waarschuwing: Input bestand {input_filename} ontbreekt in {input_dir}.")
            continue
            
        # Lees beelden in grayscale
        target_img = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        input_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        
        if target_img is None or input_img is None:
            continue
            
        # Binariseer (achtergrond 0, inkt/dots 255)
        _, t_bin = cv2.threshold(target_img, 127, 255, cv2.THRESH_BINARY_INV)
        _, i_bin = cv2.threshold(input_img, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Masker genereren
        mask = cv2.subtract(t_bin, i_bin)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Sla masker op
        cv2.imwrite(mask_path, mask)
        mask_paths.append(mask_path)
        
    # Schrijf mask flist
    mask_flist_path = os.path.join(OUTPUT_DIR, f"{split_name}_masks.flist")
    with open(mask_flist_path, 'w') as f:
        for p in mask_paths:
            f.write(p + '\n')
            
    print(f"Maskers opgeslagen in {mask_dir} en flist aangemaakt op {mask_flist_path}")

def create_flist(folder_path, output_filename):
    if not os.path.exists(folder_path):
        return
    files = sorted(glob.glob(os.path.join(folder_path, "*.png")) + glob.glob(os.path.join(folder_path, "*.jpg")))
    with open(output_filename, 'w') as f:
        for file in files:
            f.write(file + '\n')
    print(f"Gemaakt: {output_filename} ({len(files)} bestanden)")

# Voer het uit voor alle splits
for split_name, paths in splits.items():
    print(f"\n--- Verwerk {split_name} ---")
    generate_masks_and_flist(paths, split_name)
    create_flist(paths["target"], os.path.join(OUTPUT_DIR, f"{split_name}_images.flist"))
    create_flist(paths["edge"], os.path.join(OUTPUT_DIR, f"{split_name}_skeletons.flist"))

print("\nKlaar met alles!")