import numpy as np
import glob
import os

def check_fixation_files():
    files = glob.glob('output/fix_seq/*.npz')
    total = len(files)
    print(f"Total fixation files: {total}")
    
    bad_count = 0
    nan_count = 0
    zero_count = 0
    
    for i, file in enumerate(files):
        if i % 500 == 0:
            print(f"Processed {i}/{total} files...")
            
        try:
            data = np.load(file)
            seq = data['seq']
            mask = data['mask']
            bad = data.get('bad', False)
            
            if bad:
                bad_count += 1
            if np.isnan(seq).any():
                nan_count += 1
                print(f"Found NaNs in file: {os.path.basename(file)}")
            if np.all(seq == 0):
                zero_count += 1
        except Exception as e:
            print(f"Error processing {os.path.basename(file)}: {e}")
    
    print(f"\nResults:")
    print(f"Total files: {total}")
    print(f"Files marked as bad: {bad_count} ({bad_count/total*100:.2f}%)")
    print(f"Files with NaNs: {nan_count} ({nan_count/total*100:.2f}%)")
    print(f"Files with all zeros: {zero_count} ({zero_count/total*100:.2f}%)")

if __name__ == "__main__":
    check_fixation_files()
