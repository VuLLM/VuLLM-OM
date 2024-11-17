import pandas as pd
import pickle
import sys
sys.path.append('/sise/home/urizlo/VuLLM_One_Stage')
from code_files.preprocess_data import Prepare_dataset_with_only_replace_only_encoder
from collections import defaultdict
import hashlib
from dotenv import load_dotenv
import argparse
import torch
import numpy as np
import sys
from utils import Nxcode_7B
# import argparse
from datasets import Dataset
from torch.cuda.amp import autocast
# import pandas as pd
import os
from tqdm import tqdm

def process_c_functions(df, column_name):
    """
    Process C functions in a DataFrame column by splitting them into lines and
    handling special cases where lines contain only "{" or "}" characters.

    Args:
        df (pandas.DataFrame): The DataFrame containing the C functions.
        column_name (str): The name of the column containing the C functions.

    Returns:
        pandas.DataFrame: The DataFrame with the processed C functions.
    """
    for index, row in df.iterrows():
        if type(row[column_name]) == float:
            continue
        lines = row[column_name].split('\n')
        processed_lines = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line in ["{", "}"] and i > 0:
                processed_lines[-1] = processed_lines[-1] + " " + line
            else:
                processed_lines.append(lines[i])
            i += 1
        df.at[index, column_name] = '\n'.join(processed_lines)
    return df



def append_spaces_suffix_to_duplicates(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Appends spaces as suffix to duplicate lines in the specified column of the given DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame.
        column (str): The column containing the function code.

    Returns:
        pd.DataFrame: The modified DataFrame with spaces appended to duplicate lines.
    """
    modified_data = data.copy()
    for i, row in modified_data.iterrows():
        if type(row[column]) == float:
            continue
        function_code = row[column]
        lines = function_code.split('\n')
        modified_lines = []
        line_counts = defaultdict(int)  # Start all values equal to 1
        for line in lines:
            if line_counts[line.strip()] > 0:
                spaces = " " * line_counts[line.strip()]
                modified_lines.append(f"{line}{spaces}")
            else:
                modified_lines.append(line)
            line_counts[line.strip()] += 1
        modified_data.at[i, column] = '\n'.join(modified_lines)
    return modified_data



def drop_duplicates(df, nonvul_column_name):
    function_groups = {}

    for i in range(len(df)):
        nonvul = df[nonvul_column_name].iloc[i]
        nonvul = nonvul.replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")
        # Split the file content into functions (assuming functions are well-defined)
        function_hash = hashlib.sha256(nonvul.encode()).hexdigest()
        if function_hash not in function_groups:
            function_groups[function_hash] = []
        function_groups[function_hash].append(i)

    indexes_to_drop = [index for index_list in function_groups.values() if len(index_list) > 1 for index in index_list[1:]]
    df = df.drop(indexes_to_drop)
    df = df.reset_index(drop=True)
    return df



def get_data(path_testset, nonvul_column_name):
    test = pd.read_csv(path_testset)
    test = process_c_functions(test, nonvul_column_name) # maybe already done in the csv --- need to check
    test = append_spaces_suffix_to_duplicates(test, nonvul_column_name)
    test.dropna(subset=[nonvul_column_name], inplace=True)
    test = drop_duplicates(test, nonvul_column_name)
    nonvul = test[nonvul_column_name].tolist()
    return nonvul



def get_num_row(input_string, original_lines):
    """
    Get the row number of a given input string in a list of original lines.

    Args:
        input_string (str): The input string to search for.
        original_lines (list): The list of original lines to search in.

    Returns:
        int: The row number of the input string in the original lines.
    """
    row = input_string.split("<endRow>")[0][2:].lstrip()
    for i in range(len(original_lines)):
        if row == original_lines[i].lstrip():
            return i
    return -1
        


def add_rows(changes_lines, i, index_to_add, original_lines, sum_added_rows):
    """
    Add rows from `changes_lines` to `original_lines` at the specified `index_to_add`.

    Args:
        changes_lines (list): List of lines to be added.
        i (int): Current index in `changes_lines`.
        index_to_add (int): Index at which the rows should be added in `original_lines`.
        original_lines (list): List of original lines.
        sum_added_rows (int): Number of rows already added.

    Returns:
        tuple: A tuple containing the updated `original_lines`, the updated `i`, and the updated `sum_added_rows`.
    """
    start_index = i
    num = index_to_add
    while i < len(changes_lines) and index_to_add == get_num_row(changes_lines[i], original_lines):
        i += 1
    for j in range(start_index, i):
        original_lines.insert(index_to_add + 1 , "+"+changes_lines[j].split("<endRow>")[1])
        index_to_add += 1
    return original_lines, i, sum_added_rows



def delete_added_rows(changes):
    """
    Deletes the added rows from the given list of changes.

    Args:
        changes (list): The list of changes.

    Returns:
        list: The modified list with added rows removed.
    """
    i = 0
    while i < len(changes):
        if changes[i][0] == "A":
            changes.pop(i)
            continue
        i += 1
    return changes



def inject_vul(nonvul, locations, injections):
    """
    Injects vulnerabilities into the nonvul code at specified locations.

    Args:
        nonvul (str): The nonvulnerable code.
        locations (str): The locations where vulnerabilities should be injected.
        injections (str): The code to inject as vulnerabilities.

    Returns:
        str: The modified code with vulnerabilities injected.
    """
    original_lines = nonvul.split('\n')
    locations = locations.split("\n")
    i = 0

    for i in range(len(original_lines)):
        if locations[0].lstrip() == original_lines[i].lstrip() and original_lines[i].lstrip()[0] != "+":
            original_lines[i] = original_lines[i].replace(locations[0].strip(), injections[0].strip())
            if "EmptyLine" in original_lines[i]:
                original_lines[i] = original_lines[i].replace("EmptyLine", "").strip()
            locations.pop(0)
            injections.pop(0)
        if len(locations) == 0 or len(injections) == 0:
            break

    original_lines = [line.rstrip() for line in original_lines]
    final_lines = [line[1:] if line != "" and line[0] == "+" else line for line in original_lines]
    modified_func = '\n'.join(line for line in final_lines if line != "")
    return modified_func



def contain(locations, nonvul):
    """
    Check if all locations in the given string are present in the nonvul list.

    Args:
        locations (str): A string containing multiple locations separated by newline characters.
        nonvul (list): A list of non-vulnerable locations.

    Returns:
        bool: True if all locations are present in the nonvul list, False otherwise.
    """
    locations = locations.split("\n")
    for loc in locations:
        if loc.strip() not in nonvul:
            return False
    return True



def delete_pre_spaces(s):
    """
    Removes leading spaces from each line in the input string and returns the cleaned string.

    Args:
        s (str): The input string.

    Returns:
        str: The cleaned string with leading spaces removed from each line.
    """
    lines = s.split('\n')
    cleaned_lines = [line.lstrip() for line in lines]
    cleaned_lines = [line for line in cleaned_lines if line != ""]
    result = '\n'.join(cleaned_lines)
    return result



def get_locations_and_injections(mod_lines):
    locations = []
    injections = []
    i = 0
    while i < len(mod_lines) and i < len(mod_lines):
        if mod_lines[i] == "<s>":
            if mod_lines[i+1][0] != "A":
                locations.append(mod_lines[i+1])
                i += 2
                inj = []
                while i < len(mod_lines) and mod_lines[i] != "<s>":
                    inj.append(mod_lines[i])
                    i += 1
                injections.append('\n'.join(inj))
            else:
                locations.append(mod_lines[i+1].split("<endRow>")[0][2:])
                i += 2
                inj = []
                while i < len(mod_lines) and mod_lines[i] != "<s>":
                    inj.append(mod_lines[i])
                    i += 1
                injections.append('\n'.join(inj))
        # i += 1
    locations = '\n'.join(locations)
    return locations, injections


def apply_modifications(func, mods):
        lines = func.split('\n')
        modified_lines = lines.copy()
        mods = mods.split('<N>')[1:]
        same_lines = True
        
        for mod in mods:
            parts = mod.split('\n')
            target_line = parts[1].lstrip()
            replacements = [line for line in parts[2:] if line.strip() != '']
            if target_line != replacements[0].lstrip():
                same_lines = False
            found = False
            for i, line in enumerate(modified_lines):
                if line.lstrip() == target_line:
                    found = True
                    if replacements[0].strip() == 'EmptyLine':
                        modified_lines.pop(i)
                    else:
                        newline = [replace for replace in replacements]
                        if len(newline) == 1:
                            newline[0] += 'newline'
                        modified_lines[i] = '\n'.join(newline)
                    break
            
            if not found:
                return "Line not exist in the function"
        modified_lines = [line[:-7] if line.endswith('newline') else line for line in modified_lines]
        return '\n'.join(modified_lines)
    
    


def modify_functions(functions, modifications):
    num_of_problem_functions = 0
    modification_groups = [mod.strip() for mod in modifications]
    modified_functions = []
    for i, func in enumerate(functions):
        try:
            modified_func = apply_modifications(func, modification_groups[i])
            if modified_func in ["Line not exist in the function", "No change in the function"]:
                num_of_problem_functions += 1
            modified_functions.append(modified_func)
        except:
            num_of_problem_functions += 1
            continue

    return modified_functions, num_of_problem_functions



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
    gen_kwargs['num_beams'] = 1
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


    

def get_modifications(nonvul, model_path):    
    # torch.cuda.set_device(0)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    max_seq_length = 4000
    
    model, tokenizer = Nxcode_7B.create_model_and_tokenizer_one_GPU(model_path)

    test = pd.DataFrame(nonvul, columns=['nonvul'])
    test['prompt'] = test.apply(lambda row: f"""function:\n{row['nonvul']}\nInstruction:\n""", axis=1)
    test.dropna(subset=['prompt'], inplace=True)

    # test['cwe'] = test['cwe'].fillna(-1)
    # train = Dataset.from_pandas(train)
    test = Dataset.from_pandas(test)

    # Function to filter out long samples
    def filter_long_samples(example):
        inputs = tokenizer(example['prompt'], truncation=True, padding=False)
        return len(inputs['input_ids']) <= max_seq_length


    test = test.filter(filter_long_samples)
    test = test.map(lambda example: tokenizer(example['prompt'], truncation=True, max_length=max_seq_length, padding=False), batched=True)
    print("Test length: ", len(test))
    # Create train and test dataloaders
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=1, num_workers=1,shuffle=False)
    modify_instructions = []
    for step, inputs in tqdm(enumerate(test_dataloader)):
        generated_tokens, labels = prediction_step(model, inputs, tokenizer)
        gnerated_text = tokenizer.decode(generated_tokens[:-1])
        modify_instructions.append(gnerated_text)
        
    return modify_instructions, test['nonvul']
        


def main(path_testset, model_path, nonvul_column_name, output_dir):
    # load_dotenv()
    nonvul = get_data(path_testset, nonvul_column_name)
    modification_instructions, nonvul = get_modifications(nonvul, model_path)
    vuls, num_of_problem = modify_functions(nonvul, modification_instructions)
    print(num_of_problem)
    print("length of vuls: ", len(vuls))
    df = pd.DataFrame()
    df['vul'] = vuls
    df.to_csv(output_dir, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model with specific command line arguments.')
    parser.add_argument('--path_testset', type=str, default='Dataset_VulGen/vulgen_test_775_with_diff_lines_spaces.csv', help='Path to test set csv file')
    parser.add_argument('--model_path', type=str, default='saved_models/codeQwen-onlyCWE-shorter-than20/checkpoint-230956', help='Path to test set csv file')
    parser.add_argument('--nonvul_column_name', type=str, default='nonvul', help='Column name in the daframe where there is the non vulnerable fucntion')
    parser.add_argument('--output_dir', type=str, default='connected_models/generated_vul/vulgen_res.csv', help='Path to where to save the new vulnerable functions csv file')
    args = parser.parse_args()
    main(args.path_testset, args.model_path, args.nonvul_column_name, args.output_dir)
