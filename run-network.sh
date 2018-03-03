#!/bin/bash
if [[ -z "${PYTHONPATH}" ]]; then
  export PYTHONPATH="/home.stud/racinmat/GANy:/mnt/datagrid/personal/racinmat/depthEstimationNN1"
else
  export PYTHONPATH="${PYTHONPATH}:/home.stud/racinmat/GANy:/mnt/datagrid/personal/racinmat/depthEstimationNN1"
fi
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/

cd ~
source gan2/bin/activate

cd /mnt/datagrid/personal/racinmat/depthEstimationNN1
python task.py