import torch
import numpy as np
import sys
sys.path.append('/sise/home/urizlo/VuLLM_One_Stage')
from utils import Nxcode_7B
from code_files.preprocess_data import Prepare_dataset_with_only_replace_only_encoder
# import argparse
from datasets import Dataset
from torch.cuda.amp import autocast
# import pandas as pd
import os
from tqdm import tqdm
from collections import Counter as Count
import pickle

def modify_attention(data, tokenizer, target_string):
    # Tokenize the target string to get the token ids
    target_tokens = tokenizer.encode(target_string, add_special_tokens=False)
    
    # Extract tensors from the BatchEncoding object
    input_ids = torch.tensor(data['input_ids']).unsqueeze(0)
    masking_attention = torch.tensor(data['attention_mask']).unsqueeze(0)
    labels = torch.tensor(data['input_ids']).unsqueeze(0)
    
    # Find positions of the target string's tokens in the input_ids
    # Create a mask for where all the target tokens appear consecutively
    for start_index in range(input_ids.size(1) - len(target_tokens) + 1):
        if torch.all(input_ids[:, start_index:start_index + len(target_tokens)] == torch.tensor(target_tokens).to(input_ids.device)):
            # Found the target string, modify the masking_attention from the end of this string
            end_index = start_index + len(target_tokens)
            if end_index < masking_attention.size(1):
                masking_attention[:, end_index:] = 0
            break  # Assuming only one occurrence or modifying after the first occurrence only
    original_data = data.copy()
    data['input_ids'] = input_ids[:, :end_index]
    data['attention_mask'] = masking_attention[:, :end_index]
    original_data['labels'] = labels[:, end_index:]
    return data, original_data



def prediction_step(model, inputs, tokenizer):
    generate_inputs, inputs = modify_attention(inputs, tokenizer, "Instruction:\n")
    generate_inputs = {
        'input_ids': generate_inputs['input_ids'].clone().detach().to(model.device),
        'attention_mask': generate_inputs['attention_mask'].clone().detach().to(model.device)
    }
    generation_inputs = generate_inputs.copy()
    # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
    # (otherwise, it would continue generating from the padded `decoder_input_ids`)
    if (
        "labels" in generation_inputs
        and "decoder_input_ids" in generation_inputs
        and generation_inputs["labels"].shape == generation_inputs["decoder_input_ids"].shape
    ):
        generation_inputs = {
            k: v for k, v in inputs.items() if k not in ("decoder_input_ids", "decoder_attention_mask")
        }
    # Arguments for generation
    gen_kwargs = {}
    gen_kwargs['max_new_tokens'] = 512 
    gen_kwargs['pad_token_id'] = tokenizer.pad_token_id
    gen_kwargs['eos_token_id'] = tokenizer.eos_token_id
    gen_kwargs['num_beams'] = 3

    # gen_kwargs['do_sample'] = True
    # gen_kwargs['top_p'] = 0.95
        
    with autocast(dtype=torch.bfloat16):
        generated_tokens = model.generate(**generation_inputs, **gen_kwargs)
               
    generated_tokens = generated_tokens[0][len(generation_inputs['input_ids'][0]):]
    # print("len generated_tokens: ", len(generated_tokens))

    return generated_tokens, inputs['labels']


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def is_accurecy(eval_preds, tokenizer):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
        
    # Convert preds to tensor if it's a NumPy array
    if isinstance(preds, np.ndarray):
        preds = torch.tensor(preds)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Ensure labels are in numpy array format
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    decoded_labels = decoded_labels[0]

    # Accuracy
    for p, l in zip(decoded_preds, decoded_labels):
        if p == l:
            return True
    return False


    

def main():    
    # torch.cuda.set_device(0)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    checkpoint = "saved_models/vulgen-last-check/checkpoint-89112"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    max_seq_length = 4000
    
    model, tokenizer = Nxcode_7B.create_model_and_tokenizer_one_GPU(checkpoint)

    # read and tokenized data
    path_trainset = "Datasets/vulgen_datasets/vulgen_test_775.csv"
    path_testset = "Datasets/vulgen_datasets/vulgen_test_775_with_diff.csv"
    full_vulgen = False
    eos = tokenizer.eos_token
    
    _, test = Prepare_dataset_with_only_replace_only_encoder.create_datasets(path_trainset, path_testset, full_vulgen=full_vulgen)
    # train['prompt'] = train.apply(lambda row: f"""function:\n{row['inputs']}\nInstruction:\n{row['outputs']}{eos}""", axis=1)
    test['prompt'] = test.apply(lambda row: f"""function:\n{row['inputs']}\nInstruction:\n{row['outputs']}{eos}""", axis=1)
    test = test[['prompt', 'cwe']]

    test['cwe'] = test['cwe'].fillna(-1)
    # train = Dataset.from_pandas(train)
    test = Dataset.from_pandas(test)

    # Function to filter out long samples
    def filter_long_samples(example):
        inputs = tokenizer(example['prompt'], truncation=True, padding=False)
        return len(inputs['input_ids']) <= max_seq_length

    # Apply the filter
    # train = train.filter(filter_long_samples)
    test = test.filter(filter_long_samples)
    # test.reset_index(drop=True, inplace=True)
    test = test.map(lambda example: tokenizer(example['prompt'], truncation=True, max_length=4000, padding=False), batched=True)
    
    # Create train and test dataloaders
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=1, num_workers=4,shuffle=False)
    correct_indexes = []
    for step, inputs in tqdm(enumerate(test_dataloader)):
        generated_tokens, labels = prediction_step(model, inputs, tokenizer)
        if is_accurecy(([generated_tokens], labels), tokenizer):
            correct_indexes.append(step)
    print("Number of correct indexes:", len(correct_indexes))

    # Get the unique CWES from full_testset.csv that appear in the correct indexes
    unique_cwes = set()
    cwe_hist = Count()
    for index in correct_indexes:
        cwe = test[index]['cwe']
        if cwe != -1:
            unique_cwes.add(cwe)
            if cwe not in cwe_hist:
                cwe_hist[cwe] = 1
            else:
                cwe_hist[cwe] += 1
    
    # Save cwe_hist as pickle
    with open('cwe_hist.pickle', 'wb') as f:
        pickle.dump(cwe_hist, f)
    print("Accuracy:", len(correct_indexes)/len(test))
    print("Length of test:", len(test))
    print("Unique CWES:", unique_cwes)
    print("Number of unique CWES:", len(unique_cwes))
    
if __name__ == "__main__":
    main()