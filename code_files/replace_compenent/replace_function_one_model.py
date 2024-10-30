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



def drop_duplicates(df):
    function_groups = {}

    for i in range(len(df)):
        nonvul = df['nonvul'].iloc[i]
        nonvul = nonvul.replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")
        lines_after_fix = df['vul'].iloc[i]
        lines_after_fix = lines_after_fix.replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")
        # Split the file content into functions (assuming functions are well-defined)
        row = nonvul + lines_after_fix  # Change this based on your function definitions
        function_hash = hashlib.sha256(row.encode()).hexdigest()
        if function_hash not in function_groups:
            function_groups[function_hash] = []
        function_groups[function_hash].append(i)

    indexes_to_drop = [index for index_list in function_groups.values() if len(index_list) > 1 for index in index_list[1:]]
    df = df.drop(indexes_to_drop)
    df = df.reset_index(drop=True)
    return df



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



def delete_spaces(s):
    """
    Removes leading spaces from each line in the input string and returns the cleaned string.

    Args:
        s (str): The input string.

    Returns:
        str: The cleaned string with leading spaces removed from each line.
    """
    lines = s.split('\n')
    cleaned_lines = [line.strip() for line in lines]
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
        modified_func = apply_modifications(func, modification_groups[i])
        if modified_func in ["Line not exist in the function", "No change in the function"]:
            num_of_problem_functions += 1
        modified_functions.append(modified_func)

    return modified_functions, num_of_problem_functions




def get_accurecy(vuls, vul_funcs):
    """
    Calculate the accuracy of vulnerability detection.

    Args:
        vuls (list): List of generated vulnerabilities.
        vul_funcs (DataFrame): DataFrame containing vulnerability functions.

    Returns:
        list: List of indices of vulnerabilities that were not detected accurately.
    """
    count = 0
    not_good = []
    for i in range(len(vuls)):
        x = delete_spaces(vuls[i])
        y = delete_spaces(vul_funcs[i]) 
        if x == y:
            count += 1
        else:
            not_good.append(i)
    print(count/len(vuls))
    return not_good




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
        'input_ids': torch.tensor(generate_inputs['input_ids']).to(model.device),
        'attention_mask': torch.tensor(generate_inputs['attention_mask']).to(model.device)
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


    

def get_modifications(testset_path, mod_exists=False):    
    # torch.cuda.set_device(0)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    checkpoint = "saved_models/codeQwen-vulgen/checkpoint-119184"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    max_seq_length = 5000
    model, tokenizer = Nxcode_7B.create_model_and_tokenizer_one_GPU(checkpoint)
    
    # read and tokenized data
    path_trainset = "Datasets/vulgen_datasets/vulgen_test_with_diff.csv"
    path_testset = testset_path
    full_vulgen = False
    eos = tokenizer.eos_token
    
    _, test = Prepare_dataset_with_only_replace_only_encoder.create_datasets(path_trainset, path_testset, full_vulgen=full_vulgen)
    # train['prompt'] = train.apply(lambda row: f"""function:\n{row['inputs']}\nInstruction:\n{row['outputs']}{eos}""", axis=1)
    test['prompt'] = test.apply(lambda row: f"""function:\n{row['inputs']}\nInstruction:\n{row['outputs']}{eos}""", axis=1)
    
    test = test[['prompt', 'cwe', 'nonvul', 'vul', 'inputs', 'outputs']]
    empty_indexes = test[test['outputs'] == ''].index.tolist()
    with open('pickle_files/empty_indexes.pickle', 'wb') as f:
        pickle.dump(empty_indexes, f)
    test = test[test['outputs'] != '']
    test.reset_index(drop=True, inplace=True)
    test['cwe'] = test['cwe'].fillna(-1)
    # test = test[:30]
    # train = Dataset.from_pandas(train)
    # test = Dataset.from_pandas(test)

    # Function to filter out long samples
    def filter_long_samples(example):
        inputs = tokenizer(example['prompt'], truncation=True, padding=False)
        return len(inputs['input_ids']) <= max_seq_length
    
    # Function to create a boolean mask for filtering
    def create_filter_mask(df, tokenizer, max_seq_length):
        mask = []
        for index, row in df.iterrows():
            inputs = tokenizer(row['prompt'], truncation=True, padding=False)
            mask.append(len(inputs['input_ids']) <= max_seq_length)
        return mask

    # Create the boolean mask
    filter_mask = create_filter_mask(test, tokenizer, max_seq_length)

    # Get the indexes of the rows to drop
    dropped_indexes = test.index[~pd.Series(filter_mask)].tolist()
    with open('pickle_files/fillter_indexes.pickle', 'wb') as f:
        pickle.dump(dropped_indexes, f)

    test = test[pd.Series(filter_mask)]
    test.reset_index(drop=True, inplace=True)
    nonvuls = test['nonvul']
    vuls = test['vul']
    outputs = list(test['outputs'])
    outputs = [row.split(eos)[0] for row in outputs]
    df = pd.DataFrame()
    df['mods'] = outputs
    df.to_csv('code_files/replace_compenent/modify_instructions.csv', index=False)
    if not mod_exists:
        test = test.map(lambda example: tokenizer(example['prompt'], truncation=True, max_length=5000, padding=False), batched=True)
        
        # Create train and test dataloaders
        test_dataloader = torch.utils.data.DataLoader(test, batch_size=1, num_workers=4,shuffle=False)
        modify_instructions = []
        for step, inputs in tqdm(enumerate(test_dataloader)):
            generated_tokens, labels = prediction_step(model, inputs, tokenizer)
            gnerated_text = tokenizer.decode(generated_tokens[:-1])
            modify_instructions.append(gnerated_text)
        df = pd.DataFrame()
        df['mods'] = modify_instructions
        df.to_csv('code_files/replace_compenent/modify_instructions.csv', index=False)
        return modify_instructions, nonvuls, vuls
    else:
        return nonvuls, vuls
        


def main(path_testset, mod_exists=False):
    if mod_exists:
        nonvuls, vuls = get_modifications(path_testset, mod_exists)
        modification_instructions = list(pd.read_csv('code_files/replace_compenent/modify_instructions.csv')['mods'])
    else:
        modification_instructions, nonvuls, vuls = get_modifications(path_testset, mod_exists)
    new_vuls, num_of_problem = modify_functions(nonvuls, modification_instructions)
    print(num_of_problem)
    not_good = get_accurecy(new_vuls, vuls)
    print(not_good)
    print(len(not_good))
    # Save not_good as pickle
    with open('pickle_files/indexes_to_drop_fullData.pickle', 'wb') as f:
        pickle.dump(not_good, f)




if __name__ == "__main__":
    path_testset = 'Datasets/vulgen_megavul_primevul_vulpatchpairs_with_diff_after_clean.csv'
    mod_exists = True
    main(path_testset, mod_exists)
