#!/bin/bash

set -e

python3.8 -m venv epictrace_venv

source epictrace_venv/bin/activate

pip install -r requirements.txt

