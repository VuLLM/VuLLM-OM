# LineVul - Effectiveness

## Introduction

LineVul is a SOTA vulnerability detection, aimed at evaluating the effectiveness of VuLLM generated samples. This project facilitates running tests across different datasets and training configurations to benchmark performance accurately.

## Table of Contents

1. [General Information](#general-information)
3. [How to Run Tests](#how-to-run-tests)
4. [Training and Testing on New Data](#training-and-testing-on-new-data)

## General Information

- This README provides all the necessary information to run and test the LineVul project effectively.
- Explore the project on GitHub: [LineVul GitHub Repository](https://github.com/awsm-research/LineVul)

## How to Run Tests

### Download CSV files
- [Download CSV files from here](https://drive.google.com/file/d/1J5jljtPMJp3iB0YEu4ZPZwmtjshIBbQ4/view?usp=sharing)
- Move this files to this foler in this path in the project `detector_models/LineVul/code/data/train_linevul_test_reveal`

### Run Trails
The project includes a comprehensive set of scripts to run tests as described in Table 3 of the related article. In the `run_trails/imbalanced`
- `Baseline.sh`: Run the test in Table 3, column "baseline."
- `OM_VuLLM.sh`: Run the test in Table 3, column "OM_VuLLM."
- `TM_VuLLM.sh`: Run the test in Table 3, column "TM_VuLLM."
- `VGX.sh`: Run the test in Table 3, column "VGX."
- `Syn.sh`: Run the test in Table 3, column "syn."
- `Wild.sh`: Run the test in Table 3, column "OM VuLLM Wild."

## Training and Testing on New Data

To train and test the model on new data, follow these steps:
   - Create new sh file like 'detector_models/LineVul/run_trails/balanced/Baseline.sh'
   - Change this 3 fileds with path to new data set.
      -   --train_data_file
      -   --eval_data_file
      -   --test_data_file


