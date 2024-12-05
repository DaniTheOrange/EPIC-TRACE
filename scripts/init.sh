#!/bin/bash

set -e

# Create a virtual environment and install the required packages for EPIC-TRACE
python3.8 -m venv epictrace_venv

source epictrace_venv/bin/activate

pip install -r requirements.txt
# for newer GPU might need
# pip install torch==1.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# create separate virtual environment for ProtBERT embeddings
cd protBERT

python3.8 -m venv protbert_venv

source protbert_venv/bin/activate

pip install -r protbert_requirements.txt

# for new GPUs
# pip install torch==1.7.0+cu110 -f https://download.pytorch.org/whl/torch_stable.html


# return to the root directory
cd ..
source epictrace_venv/bin/activate