# RL Dataset Creation

## Instructions

1. Copy the `output` dataset from `Stage1_Data_Pipeline`
2. Run `score.pbs` on Katana HPC.
3. Run `filter.py` on Katana HPC or locally with the desired token limit.
4. Save the `output_filtered` dataset and pass to `Stage1_SFT_Dataset_Creation`/`Stage1_RL_Dataset_Creation` or `Stage3_Forecast_Benchmark` to create a training dataset or as an evaluation dataset.
