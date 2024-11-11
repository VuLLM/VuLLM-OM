# VuLLM-OM

## Introduction

VuLLM is tool for injecting vulnerabilities to C-code functions. VuLLM is utilizing Code LLM (CodeQwen1.5-7B)to learn the specific text modification instructions for generating the vulnerable C-Code function.

## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Contact](#Contact)

## Project Structure
```bash
VuLLM_One_Stage/
├── .env
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
│   │   └── __pycache__/
│   │       ├── Prepare_dataset_with_only_replace.cpython-310.pyc
│   │       ├── Prepare_dataset_with_only_replace_codeT5.cpython-310.pyc
│   │       ├── Prepare_dataset_with_only_replace_mistral.cpython-310.pyc
│   │       └── Prepare_dataset_with_only_replace_only_encoder.cpython-310.pyc
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


## Usage

### Download VulGen Dataset from Google Drive

Download files from here: https://drive.google.com/file/d/1hzq_i01IqKSIaGcKEpkw3OHUlD7neScT/view?usp=drive_link

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


### Get Results of Acuurecy in Table 1 - Infernce all process (localization, injection models and replacment component).

#### Infernce Localization model on 775 samples from VulGen test set

Run `connected_models/pipeline_localization_vulgen.py`<br>
Arguments:
- `path_testset` (str): Path to test set csv file.
- `model_huggingface_path` (str): Path to load lora adapters form huggingface.
- `all_vulgen` (bool): Is the test set is the whole test set of VulGen?
- `output_dir` (srt): Path to where to save the loclization csv file.

To run on 775 samples from VulGen, path_testset: `Dataset_VulGen/vulgen_test_775_with_diff_lines_spaces.csv`.

#### Example for running

`python connected_models/pipeline_localization_vulgen.py --path_testset Dataset_VulGen/vulgen_test_775_with_diff_lines_spaces.csv --model_huggingface_path urizlo/CodeT5-6B-inject-acc0.6793-dropout0.05-r64-lr6e-5-epochs30-dropDuplicate --all_vulgen True --output_dir connected_models/localization_results/vulgen_res_loc.csv`

#### Infernce Injection model on 775 samples from VulGen test set

Run `connected_models/pipeline_injection_vulgen.py`<br>
Arguments:
- `path_testset` (str): Path to test set csv file.
- `model_huggingface_path` (str): Path to load lora adapters form huggingface.
- `all_vulgen` (bool): Is the test set is the whole test set of VulGen?
- `path_res_local` (str): Path to localization results of the testset.
- `output_dir` (srt): Path to where to save the loclization csv file.

#### Example for running

`python connected_models/pipeline_injection_vulgen.py --path_testset Dataset_VulGen/vulgen_test_775_with_diff_lines_spaces.csv --model_huggingface_path urizlo/CodeT5-6B-local-acc0.5271-dropout0.05-r64-lr6e-5-epochs30-dropDuplicate --all_vulgen True --path_res_local connected_models/localization_results/vulgen_res_loc.csv --output_dir connected_models/injection_results/vulgen_res_inj.csv`

#### Operate Replacment component

Run `connected_models/replace_function_with_line_spaces_vulgen.py`<br>
Arguments:
- `path_testset` (str): Path to test set csv file.
- `path_res_local` (str): Path to localization results of the testset.
- `path_res_inj` (str): Path to injection results of the testset.
- `all_vulgen` (bool): Is the test set is the whole test set of VulGen?
- `output_dir` (srt): Path to where to save the new vulnerable functions csv file.

#### Example for running

`python connected_models/replace_function_with_line_spaces_vulgen.py --path_testset Dataset_VulGen/vulgen_test_775_with_diff_lines_spaces.csv --path_res_local connected_models/localization_results/vulgen_res_loc.csv  --path_res_inj connected_models/injection_results/vulgen_res_inj.csv --all_vulgen True --output_dir connected_models/genrated_vul/vulgen_res.csv`

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


### Get Results of Effectiveness in Table 2.

#### LineVul

Read README.md for this folder in this path `detector_models/LineVul/VuLLM_README.md`

#### Devign

Read README.md for this folder in this path `detector_models/Devign/VuLLM_README.md`


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Contact

Uri Zlotkin - uri@zlotkin.com

