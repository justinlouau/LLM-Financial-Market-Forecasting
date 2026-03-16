# LLM Financial Market Forecasting

This repository provides source code for the thesis *Leveraging Time Series Multimodal Large Language Models for Probabilistic Forecasting of Financial Markets* alongside instructions on how to replicate our experimental results.

## System Requirements

- This research was performed on the [UNSW Katana High Performance Computing Cluster](https://docs.restech.unsw.edu.au/). For our implementation, we used 2x H200s for model training and 1x H200 for inference/benchmarking tasks.
- Python scripts with an associated `.pbs` are intended to be run on Katana or another HPC Cluster. See HPC_Setup.md for more information.
- Python scripts without an associated `.pbs` file can be run on workstations using [pdm package manager](https://pdm-project.org/en/latest/). Begin by installing `pdm` and running `pdm install`, then following the folder specific instructions.

## Getting Started

This repository is grouped into multiple steps and stages. Detailed instructions are available inside each individual folder.

**Stage 1: Data Pipeline and Dataset Creation**

- Stage1_Data_Pipeline: This is a pipeline which retrieves S&P 500 stock data, financial reports, and news data as the foundational data for training and evaluation.
- Stage1_SFT_Dataset_Creation: This contains the source code used to create our Supervised Fine Tuning (SFT) Dataset.
- Stage1_RL_Dataset_Creation: This contains the source code used to create our Reinforcement Learning (RL) Dataset.

**Stage 2: Training**

- Stage2_ChatTS_Training: This contains the source code used to train the SFT and RL stages of the model.

**Stage 3: Benchmark**

- Stage3_Forecast_Benchmark: This contains the source code used to benchmark our model.

**Stage 4: Evaluation**

- Stage4_Forecast_Evaluation: This contains the source code used to evaluate our model.

**Miscellaneous**

- FinArena_Benchmark: This contains the benchmark code used in our research proposal.

## Contact

For any questions, issues, or other concerns, please raise an issue [here](https://github.com/justinlouau/LLM-Financial-Market-Forecasting/issues). 

The dataset and full report are avaliable on request, and is anticipated to be released later in 2026.
