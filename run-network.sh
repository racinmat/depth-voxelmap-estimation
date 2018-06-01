#!/bin/bash

ml Pillow/5.0.0-goolfc-2018a-Python-3.6.4
ml TensorFlow/1.7.0-goolfc-2018a-Python-3.6.4

export PYTHONPATH="${PYTHONPATH}:/mnt/datagrid/personal/racinmat"
export PYTHONPATH="${PYTHONPATH}:/mnt/datagrid/personal/racinmat/GTAVisionExport_postprocessing"

cd /mnt/datagrid/personal/racinmat/depthEstimationNN1
python3 task.py