# SFT Dataset Creation

## Instructions

1. Copy the `output_filtered` dataset from `Stage1_Dataset_Preparation`
2. Run `1_prepare_training_dataset.py` locally or on Katana HPC.
3. Run `2_generate_training_dataset.pbs` on Katana HPC.
4. Run `3_clean_training_set.py` locally or on Katana HPC.
5. Run `4_create_training_set.py` locally or on Katana HPC.
6. Save the `.jsonl` output file and pass to `Stage2_ChatTS_Training/train_sft_lora.pbs` to perform training
