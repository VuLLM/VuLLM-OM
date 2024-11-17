# Deivgn - Effectiveness

## Introduction

Deivgn is a SOTA vulnerability detection, aimed at evaluating the effectiveness of VuLLM generated samples. This project facilitates running tests across different datasets and training configurations to benchmark performance accurately.

## Table of Contents

1. [General Information](#general-information)
2. [Installation](#installation)
3. [How to Run Tests](#how-to-run-tests)
4. [Training and Testing on New Data](#training-and-testing-on-new-data)
5. [Dependencies](#dependencies)
7. [License](#license)

## General Information

- This README provides all the necessary information to run and test the Deivgn project effectively.
- A new `requirements.txt` file has been added to manage project dependencies.
- Explore the project on GitHub: [Deivgn GitHub Repository](https://github.com/epicosy/devign)

## Installation

Before running the project, ensure all dependencies are installed. Run the following command in the project's root directory: pip install -r requirements.txt

## How to Run Tests

- [Download all devign data](https://drive.google.com/file/d/1uNn4iN7cEZbUnTLxI8KVK4Mj3dw6kXUv/view?usp=sharing)
- Unzip this ZIP file and locate it in this path: `detector_models/devign/data`

The project includes a comprehensive set of scripts to run tests as described in Table 3 of the related article. In the `run_trails\imbalanced`

- `Baseline.sh`: Runs the test in Table 3, column "Baseline."
- `OM_VuLLM.sh`: Runs the test in Table 3, column "OM VuLLM."
- `TM_VuLLM.sh`: Runs the test in Table 3, column "TM VuLLM."
- - `train_VGX.sh`: Runs the test in Table 3, column "VGX."
- `train_syn.sh`: Runs the test in Table 2, column "syn."
- `train_wild.sh`: Runs the test in Table 2, column "OM VuLLM Wild."

## Training and Testing on New Data

To train and test the model on new data, follow these steps:

1. **Prepare New Data:**
   - In `data/raw`, create a `.json` file with new data in the following format: `{"code": "C function: str", "target": 0/1: int, "test": 0}`.
   - `target` key value `0` means non-vulnerable function and `1` means vulnerable function.
   - In `test`, set `0` if you want this function in the train set and `1` for it to be in the test set.

2. **Generate CPG Representation:**
   - Run `main.py -c` to generate CPG representations (`x_cpg.bin`, `x_cpg.json`, `x_cpg.json`) in `data/cpg`.

3. **Update Number Variable:**
   - Run `changes_numbers.py` and set the `number` variable to a value that does not exist in all inputs folders.

4. **Generate Input Data:**
   - Run `main.py -e` to generate `x_cpg_input.pkl` in `data/input`.

5. **Prepare Training and Testing Data:**
   - Place all `.pkl` files that you want to use for training and testing in `data/input`.

6. **Run Training/Testing:**
   - Execute `main.py -p/-pS --num_epochs <number_of_epochs>` to start the training/testing process.

## Dependencies

Describe in README.md


## License

This project is licensed under the [MIT License](LICENSE.txt).

