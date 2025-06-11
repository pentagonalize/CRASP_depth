#!/bin/bash
# This script removes all .o files in the current directory and its subdirectories.

find . -name "andy_depth_redo.*" -print0 | xargs -0 rm -f

# Empty L3 to L12 folders in models subfolders
for layer_dir in *_layer; do
  if [ -d "$layer_dir/models" ]; then
    for model_dir in "$layer_dir/models"/L[3-9] "$layer_dir/models"/L1[0-2]; do
      if [ -d "$model_dir" ]; then
        # Use find and xargs to remove all files and directories *within* the L3-L12 directories
        find "$model_dir" -mindepth 1 -print0 | xargs -0 rm -rf 
        echo "Emptied directory: $model_dir" # added echo
      fi
    done
  fi
done

# Remove logfiles
rm *.o*
