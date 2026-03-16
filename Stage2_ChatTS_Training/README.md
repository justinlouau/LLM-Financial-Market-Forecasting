# ChatTS Training

## SFT Training Instructions

1. Copy the training dataset from `Stage2_SFT_Dataset_Creation` or `Stage2_RL_Dataset_Creation`
2. Run `train_sft_lora.py` on Katana HPC.
3. Prepare the base model and edit the filepaths in `merge_lora.pbs`
4. Run `merge_lora.pbs` on Katana HPC and save the merged model

## RL Training Instructions

1. Copy the training dataset from `Stage2_RL_Dataset_Creation`
2. Run `train_sft_lora.py` on Katana HPC.
3. Prepare the base model and edit the filepaths in `merge_lora.pbs`
4. Run `merge_lora.pbs` on Katana HPC and save the merged model
