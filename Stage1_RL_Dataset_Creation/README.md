# RL Dataset Creation

## Instructions

1. Copy the `output_filtered` dataset from `Stage1_Dataset_Preparation`
2. Run `rf_training.pbs` on Katana HPC.
3. Run `create_rl_training_set.py` on the DPO Output Pairs.
4. Save the `.jsonl` output file and pass to `Stage2_ChatTS_Training/train_rl_lora.pbs` to perform training
