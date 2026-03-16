# HPC Setup

This page details how to setup the environment for an HPC Cluster to be able to run PBS files and their associated scripts.

```shell
module purge
cd /home/REPLACE_WITH_USER_ACCOUNT/
module load python/3.11.3
module load cuda/12.8.0
module load gcc/12.2.0

python3 -m venv .venv
source .venv/bin/activate

pip3 install --upgrade pip
pip3 install wheel

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128

pip3 install ninja
pip3 install deepspeed
pip3 install transformers accelerate datasets pandas peft bitsandbytes pydantic lm-format-enforcer einops sentencepiece protobuf trl

MAX_JOBS=6 pip3 install --cache-dir /srv/scratch/REPLACE_WITH_USER_ACCOUNT/.cache flash_attn --no-cache-dir --no-build-isolation
```