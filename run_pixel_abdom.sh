#!/bin/bash
python3 /workspace/uas_mood/predict_folder.py -i $1 -o $2 -m 'pixel' --model_ckpt '/workspace/uas_mood/model_ckpts/abdom_256_polygons_3views_3slices_train250_last.ckpt'
