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

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Contact


