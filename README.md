# VuLLM-OM

## Introduction

VuLLM One Model (OM) is tool for injecting vulnerabilities to C-code functions. VuLLM OM is utilizing Code LLM (CodeQwen1.5-7B) to learn the specific text modification instructions for generating the vulnerable C-Code function.

## Table of Contents
- [Project Structure](#Project-Structure)
- [Installation](#Installation)
- [CSV Datasets](#CSV-Datasets)
- [Download Models](#Download-Models)
- [Reproduce Tables Results](#Reproduce-Tables-Results)
- [Usage](#Usage)
- [License](#License)
- [Contact](#Contact)

## Project Structure
```bash
VuLLM_One_Stage/
├── LICENSE
├── README.md
├── accelerate_config_files/
│   ├── deepspeed.json
│   └── deepspeed_stage2.yaml
├── code_files/
│   ├── Save_and_load_hub.ipynb
│   ├── check_cwes/
│   │   └── cwes.py
│   ├── fine_tuning/
│   │   └── CodeQwen/
│   │       ├── Fine_tuning_one_GPU_only_replace.py
│   │       └── multi_gpus.py
│   ├── generate_vul_functions/
│   │   ├── edit_data.py
│   │   ├── gen_vul.py
│   │   ├── pipeline_vulgen.py
│   │   ├── pipline_vulgen_only_repalce.py
│   │   └── primevul_test.jsonl
│   ├── preprocess_data/
│   │   ├── Prepare_dataset_with_line_spaces.py
│   │   ├── Prepare_dataset_with_only_replace_only_encoder.py
│   └── replace_compenent/
│       ├── modify_instructions.csv
│       ├── replace_function_one_model.py
│       ├── replace_function_with_line_spaces_vulgen.py
│       └── replace_function_with_only_replace.py
├── cwe_hist_TM.png
├── cwes.pickle
├── detector_models/
├── plots.py
├── requirements.txt
├── utils/
│   ├── CodeT5p_6B.py
│   ├── Create_lora.py
│   ├── Create_lora_Llama_3_8B.py
│   ├── Create_lora_mistral.py
│   ├── Create_lora_starCoder.py
│   ├── Custom_SFTTrainer.py
│   ├── Custom_trainer.py
│   ├── DeepSeek_7B.py
│   ├── Llama_3_8B.py
│   ├── Mistral_7B.py
│   ├── Nxcode_7B.py
│   ├── StarCoder2_7B.py                                      # Project dependencies
```

## Installation
- python 3.9+<br>
To install the necessary dependencies, run the following command:
```sh
pip install -r requirements.txt
```
### Hardware and Software requirements
- Linux OS
- GPU with 48 GB RAM

  
## CSV Datasets

### VulGen Dataset Samples Subsets
- Download VulGen 8,586 samples Train set: [VulGen Train set](https://drive.google.com/file/d/1jz8uRy475PpGngWAl6k8o6Vhvm60LYEk/view?usp=sharing)
<a id="vulgen-testset"></a>
- Download VulGen 775 samples Test set: [VulGen Test set](https://drive.google.com/file/d/1ZxGnRa8VmSR-EKfBxwcdbeOuYfUIxSbk/view?usp=sharing)
- Download VulGen 479 Test set that used for Effectivness: [Samples Used for Effectivness](https://drive.google.com/file/d/14emWYvRm-M3_jdbdcBBs8QOtJGf-OtlJ/view?usp=sharing)
### VuLLM Dataset Samples Subsets
- Download VuLLM 31,679 samples Train set: [VuLLM Dataset](https://drive.google.com/file/d/1I9t6s5bwNHTaGs7zdfeBZLFv7YpeGD2Q/view?usp=sharing)
- Download VuLLM 7,166 samples Train set shorter than 5: [VuLLM Train set Shorter Than 5](https://drive.google.com/file/d/1G7sLvrBSqg5WW96Iqhj-VH2u3AquqvpD/view?usp=sharing)
- Download VuLLM 289 samples Test set shorter than 5: [VuLLM Test set Shorter Than 5](https://drive.google.com/file/d/1qh3EdzoPGdVex3183EIIrFDgOptXcXD9/view?usp=sharing)
- Download VuLLM 14,420 samples Train set shorter than 10: [VuLLM Train set Shorter Than 10](https://drive.google.com/file/d/1oNK2ejkl83-Jo56ZlTpyKjdl9AFkxgrC/view?usp=sharing)
- Download VuLLM 637 samples Test set shorter than 10: [VuLLM Test set Shorter Than 10](https://drive.google.com/file/d/16CnmQwUkMiUUDSYSxNtVeXZt2R-Yi1ez/view?usp=sharing)
- Download VuLLM 23,069 samples Train set shorter than 20: [VuLLM Train set Shorter Than 20](https://drive.google.com/file/d/1Wu-pXk6QdMgQgZN8ZwGBysx-KYgPWRo4/view?usp=sharing)
- Download VuLLM 1,064 samples Test set shorter than 20: [VuLLM Test set Shorter Than 20](https://drive.google.com/file/d/1teLkEGhPU_N16idmDkm87M1cylrahi8l/view?usp=sharing)
- Download VuLLM 27,318 samples Train set shorter than 30: [VuLLM Train set Shorter Than 30](https://drive.google.com/file/d/15dSPyA9RjPXkRE6Nxhj-EKLZHMhPhvH1/view?usp=sharing)
- Download VuLLM 1,274 samples Test set shorter than 30: [VuLLM Test set Shorter Than 30](https://drive.google.com/file/d/1vp2isYcva1_nTjGMdVS0lEKhVEAVh3_y/view?usp=sharing)


  ## Download Models
<a id="model-table1"></a>
  - Download Fine-tuned model for table 1 (fine-tuned on VulGen dataset):[VuLLM fine-tuned on VulGen](https://drive.google.com/file/d/145QiDo1MI60ewsLbRFwXJWAycWLz9uns/view?usp=sharing)
  - Download Fine-tuned model on VuLLM Dataset shorter than 5:[VuLLM shorter than 5](https://drive.google.com/file/d/1tu3BTaPrKnkB8fdKYfFMwTMSSfxAT1Uy/view?usp=sharing)
  - Download Fine-tuned model on VuLLM Dataset shorter than 10:[VuLLM shorter than 10](https://drive.google.com/file/d/1grzsfX1xBcjc8C_I8EAKbwJJXfJ02Awd/view?usp=sharing)
  - Download Fine-tuned model on VuLLM Dataset shorter than 20:[VuLLM shorter than 20](https://drive.google.com/file/d/1JeRjE3HeOg4E95efIfl0xzUvGuNVeIjL/view?usp=sharing)
  - Download Fine-tuned model on VuLLM Dataset shorter than 30:[VuLLM shorter than 30](https://drive.google.com/file/d/1-jpCV-lZx7CEAEmSnUSwyJNIe_XHaedt/view?usp=sharing)


## Reproduce Tables Results

### Get Results of Comperative Accuracy in Table 1 - Inference VuLLM OM.
1. Download [VulGen 775 samples Test set](#vulgen-testset) and save it.
2. Download [Fine-tuned model for table 1](#model-table1) and save it.
3. Open `reproduce/table1.sh` change the following Arguments:
- `path_testset` (str): Path to VulGen 775 samples Test set.
- `model_path` (str): Path to model for table 1.
- `output_dir` (srt): Path to where to save the generated vulnerable functions from 775 samples.
3. Run `sh reproduce/tables.sh`

### Get Results of Relative Accurecy in Table 2
1. Download the testsets of VuLLM Dataset Samples Subsets.
- In the CSVs Make sure that the column with vulneravle functions called `vul` and with the non-vulnerable functions called `nonvul`
3. Download 4 models for shorter than 5,10,20,30.
4. Open `reproduce/table1.sh` change the following Arguments:
- `path_testset` (str): Path to currect testset.
- `model_path` (str): Path to currect model.
- `output_dir` (srt): Path to where to save the generated vulnerable functions.
3. Run `sh reproduce/tables.sh`

### Get Resutls of Effectivness in Table 3
#### LineVul
- Read README.md for this folder in this path `detector_models/LineVul/VuLLM_README.md`
#### Devign
- Read README.md for this folder in this path `detector_models/Devign/VuLLM_README.md`

---------------------------------------------------------------------------------------------------------------------------
## Usage

### Fine-tuning the Localization Model

#### Running the Scripts
To fine-tune the CodeT5+ 6B model, use one of the following scripts depending on your setup:
- For a single GPU setup, run `localization_model/CodeT5p/Fine_tuning_one_GPU.py`.
- For multi-GPU setups using accelerators, run `localization_model/CodeT5p/Fine_tuning_accelerator.py` with this command in terminal: `NCCL_P2P_DISABLE='1' OMP_NUM_THREADS='1' accelerate launch --config_file accelerate_config_files/deepspeed_stage2.yaml localization_model/CodeT5p/Fine_tuning_accelerator.py --path_trainset Dataset_VulGen/vulgen_train_with_diff_lines_spaces.csv --path_testset Dataset_VulGen/vulgen_test_with_diff_lines_spaces.csv --is_vulgen True --output_dir saved_models --learning_rate 5e-5 --batch_size_per_device 1 --epochs 30 --generation_num_beams 1`, you can change the arguments.

#### Running with Neptune
Create `.env` file for this 2 lines if you want to use neptune:
- NEPTUNE_API_TOKEN = os.environ.get("NEPTUNE_API_TOKEN")
- NEPTUNE_PROJECT = os.environ.get("NEPTUNE_PROJECT")

If you do not want running with Neptune delete this 2 lines, and change `report_to=None` argument in Seq2SeqTrainingArguments.

#### Fine-tuning on VulGen Dataset

- If you want to use VulGen Dataset `is_vulgen=True`
- If not 
    - 1.create CSV file with 2 column: `vul` and `nonvul` are pairs of the same function one is the vulnerable form `vul` and it corresponded fix version is `nonvul`
    - 2.Use `Exploratory data analysis/add_diff_lines.py` python file to create csv file with this columns: `vul`, `nonvul`, `lines_after_fix`

#### Arguments
When running the fine-tuning scripts, the following arguments can be specified:
- `path_trainset` (str): Path to train set csv file.
- `path_testset` (str): Path to test set csv file.
- `is_vulgen` (bool): Is the training set and the test set are from the vulgen dataset.
- `output_dir` (str): The directory where the fine-tuned model will be saved, both during and after the fine-tuning process.
- `learning_rate` (float): The learning rate for the fine-tuning model.
- `batch_size_per_device` (int): The batch size per GPU device.
- `epochs` (int): The number of epochs to run for the fine-tuning process.
- `generation_num_beams` (int): The number of beams to use during the generation phase in evaluation.

#### Example for running
`python localization_model/CodeT5p/Fine_tuning_one_GPU.py --path_trainset Dataset_VulGen/vulgen_train_with_diff_lines_spaces.csv --path_testset Dataset_VulGen/vulgen_test_with_diff_lines_spaces.csv --is_vulgen True --output_dir saved_models --learning_rate 5e-5 --batch_size_per_device 1 --epochs 30 --generation_num_beams 1`

### Fine-tuning the Injection Model

#### Running the Scripts
To fine-tune the CodeT5+ 6B model, use one of the following scripts depending on your setup:
- For a single GPU setup, run `injection_model/CodeT5p/Fine_tuning_one_GPU.py`.
- For multi-GPU setups using accelerators, run `injection_model/CodeT5p/Fine_tuning_accelerator.py` with this command in terminal: `command = "NCCL_P2P_DISABLE='1' OMP_NUM_THREADS='1' accelerate launch --config_file accelerate_config_files/deepspeed_stage2.yaml injection_model/Fine_tuning_accelerator.py --path_trainset Dataset_VulGen/vulgen_train_with_diff_lines_spaces.csv --path_testset Dataset_VulGen/vulgen_test_with_diff_lines_spaces.csv --is_vulgen True --output_dir saved_models --learning_rate 5e-5 --batch_size_per_device 1 --epochs 30 --generation_num_beams 1`, you can change the arguments.

#### Running with Neptune
Create `.env` file for this 2 lines if you want to use neptune:
- NEPTUNE_API_TOKEN = os.environ.get("NEPTUNE_API_TOKEN")
- NEPTUNE_PROJECT = os.environ.get("NEPTUNE_PROJECT")

If you do not want running with Neptune delete this 2 lines, and change `report_to` argument in Seq2SeqTrainingArguments.

#### Fine-tuning on VulGen Dataset or Custom Dataset

- If you want to use VulGen Dataset argument `is_vulgen=True`
- If not 
    - 1.create CSV file with 2 column: `vul` and `nonvul` are pairs of the same function one is the vulnerable form `vul` and it corresponded fix version is `nonvul`
    - 2.Use `Exploratory data analysis/add_diff_lines.py` python file to create csv file with this columns: `vul`, `nonvul`, `lines_after_fix`

#### Arguments
When running the fine-tuning scripts, the following arguments can be specified:
- `path_trainset` (str): Path to train set csv file.
- `path_testset` (str): Path to test set csv file.
- `is_vulgen` (bool): Is the training set and the test set are from the vulgen dataset.
- `output_dir` (str): The directory where the fine-tuned model will be saved, both during and after the fine-tuning process.
- `learning_rate` (float): The learning rate for the fine-tuning model.
- `batch_size_per_device` (int): The batch size per GPU device.
- `epochs` (int): The number of epochs to run for the fine-tuning process.
- `generation_num_beams` (int): The number of beams to use during the generation phase in evaluation.

#### Example for running
`python localization_model/CodeT5p/Fine_tuning_one_GPU.py --path_trainset Dataset_VulGen/vulgen_train_with_diff_lines_spaces.csv --path_testset Dataset_VulGen/vulgen_test_with_diff_lines_spaces.csv --is_vulgen True --output_dir saved_models --learning_rate 5e-5 --batch_size_per_device 1 --epochs 30 --generation_num_beams 1`


### Infernce all process (localization, injection models and replacment component) for custom dataset.

#### Infernce Localization model on custom dataset

Run `connected_models/pipeline_localization_custom.py`<br>
Arguments:
- `path_testset` (str): Path to test set csv file.
- `model_huggingface_path` (str): Path to load lora adapters form huggingface.
- `output_dir` (srt): Path to where to save the loclization csv file.

#### Example for running

`python connected_models/pipeline_localization_custom.py --path_testset Dataset_VulGen/vulgen_test_775_with_diff_lines_spaces.csv --model_huggingface_path urizlo/CodeT5-6B-local-acc0.5271-dropout0.05-r64-lr6e-5-epochs30-dropDuplicate --all_vulgen True --output_dir connected_models/localization_results/custom_res_loc.csv`

#### Infernce Injection model on custom dataset

Run `connected_models/pipeline_injection_custom.py`<br>
Arguments:
- `path_testset` (str): Path to test set csv file.
- `model_huggingface_path` (str): Path to load lora adapters form huggingface.
- `path_res_local` (str): Path to localization results of the testset.
- `output_dir` (srt): Path to where to save the loclization csv file.

#### Example for running

`python connected_models/pipeline_localization_custom.py --path_testset Dataset_VulGen/vulgen_test_775_with_diff_lines_spaces.csv --model_huggingface_path urizlo/CodeT5-6B-local-acc0.5271-dropout0.05-r64-lr6e-5-epochs30-dropDuplicate --all_vulgen True --path_res_local connected_models/localization_results/res_loc.csv --output_dir connected_models/injection_results/custom_res_inj.csv`

#### Operate Replacment component

Run `connected_models/replace_function_with_line_spaces_custom_dataset.py`<br>
Arguments:
- `path_testset` (str): Path to test set csv file.
- `path_res_local` (str): Path to localization results of the testset.
- `path_res_inj` (str): Path to injection results of the testset.
- `output_dir` (srt): Path to where to save the new vulnerable functions csv file.

#### Example for running

`python connected_models/replace_function_with_line_spaces_custom_dataset.py --path_testset Dataset_VulGen/vulgen_test_775_with_diff_lines_spaces.csv --path_res_local connected_models/localization_results/custom_res_loc.csv  --path_res_inj connected_models/injection_results/custom_res_inj.csv --all_vulgen True --output_dir connected_models/genrated_vul/custom_res.csv`



## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Contact


