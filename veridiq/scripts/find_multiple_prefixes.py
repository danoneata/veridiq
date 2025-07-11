import os
from collections import defaultdict
import re

def find_subfolders_with_multiple_prefixes(root_folder, target_depth=3):
    subfolders_with_multiple_prefixes = []
    
    for root, dirs, files in os.walk(root_folder):
        # Calculate the depth relative to root_folder
        depth = root.replace(root_folder, '').count(os.sep)
        
        # Only process folders at the target depth
        if depth == target_depth:
            prefixes = set()
            
            for filename in files:
                if filename.endswith('.png'):
                    # Extract prefix (everything before the last underscore and number)
                    match = re.match(r'^(.+)\.mp4_frame_\d+\.png$', filename)
                    if match:
                        prefix = match.group(1)
                        prefixes.add(prefix)
            
            if len(prefixes) > 1:
                relative_path = os.path.relpath(root, root_folder)
                subfolders_with_multiple_prefixes.append(relative_path)
                print(f"Subfolder '{relative_path}' has prefixes: {sorted(prefixes)}")
    
    return subfolders_with_multiple_prefixes

# Usage
root_folder = "/data/audio-video-deepfake/FSFM_face/real+fake_data_face/train"
result = find_subfolders_with_multiple_prefixes(root_folder)