from numpy import delete
import pandas as pd
import difflib
import pickle
from datasets import Dataset
import re
from collections import defaultdict
from transformers import AutoTokenizer
import hashlib
from matplotlib import pyplot as plt
import hashlib
from pycparser import c_parser, c_ast, parse_file
import pycparser

def last_diff(diff, changes):
    """
    Returns the  difference of the last line of the code, in the given diff list and appends it to the changes list.

    Parameters:
    diff (list): The list of differences.
    changes (list): The list to append the last difference to.

    Returns:
    list: The updated changes list.
    """
    i = len(diff) - 1
    if diff[i].startswith('- '):
        changes.append("R " + diff[i][2:])
    elif diff[i].startswith('+ '):
        # line present in f2 but not in f1
        changes.append(f"A:{diff[max(0,i - 1)][2:]}<endRow> {diff[i][2:]}")
    return changes




def get_index_to_add(diff, i):
    """
    Get the index to add based on the given diff and current index.

    Args:
        diff (list): The diff list.
        i (int): The current index.

    Returns:
        int: The index to add.

    """
    index_to_add = i
    while index_to_add >= 0 and diff[index_to_add][0] != ' ':
        index_to_add -= 1
    return index_to_add



def compare_functions(f1, f2):
    """
    Compare two functions and identify the changes made between them.

    Args:
        f1 (str): The first function as a string.
        f2 (str): The second function as a string.

    Returns:
        list: A list of changes made between the two functions.
    """
    changes = []
    f1_lines = f1.splitlines()
    f2_lines = f2.splitlines()
    d = difflib.Differ()
    diff = list(d.compare(f1_lines, f2_lines))
    amount_of_added_lines = 0
    i = 0
    while i < len(diff):
        if diff[i].startswith("+") or diff[i].startswith("?"):
            amount_of_added_lines += 1
        if i + 1 == len(diff):
            changes = last_diff(diff, changes)
            return changes
        if diff[i].startswith('  '):
            # unchanged line
            i += 1
            continue
        elif diff[i].startswith('- '):
            # line present in f1 but not in f2
            changes.append("R " + diff[i][2:])
            isAdded = False
            i += 1
            while not (diff[i].startswith('  ') or  diff[i].startswith('- ')):
                if diff[i].startswith("+") or diff[i].startswith("?"):
                    amount_of_added_lines += 1
                if diff[i].startswith('+ '):                 
                    changes.append("A " + diff[i][2:])
                    isAdded = True
                i += 1
                if i == len(diff):
                    return changes
            if not isAdded:
                changes.append("A " + "EmptyLine")
            i -= 1
        elif diff[i].startswith('+ '):
            # line present in f2 but not in f1
            if i == 0:
                changes.append(f"A:{diff[1][2:]}<endRow>{diff[i][2:]}")
            else:
                if diff[i-1].startswith('+ '):
                    index_to_add = get_index_to_add(diff, i)
                    changes.append(f"A:{diff[index_to_add][2:]}<endRow>{diff[i][2:]}")
                else:
                    changes.append(f"A:{diff[max(0,i - 1)][2:]}<endRow>{diff[i][2:]}")
        i += 1
    return changes
    


# get one edit
def one_edit(s, index):
    """
    Extracts edits from a list of strings starting from a given index.

    Args:
        s (list): List of strings.
        index (int): Starting index.

    Returns:
        tuple: A tuple containing the updated index and the extracted edits as a string.
    """
    edits = ""
    count = 0
    while s[index+1][0] == "A" and s[index+1][1] != ":":
        edits += (s[index+1][1:]) +"\n"
        count += 1
        index += 1
        if index+1 == len(s):
            break
    return index, edits



def remove_number_suffix(s):
    return re.sub(r"_\d{1,3}$", "", s)



def clear_edits(edits):
    """
    Removes redundant edits from the given list of edits.

    Args:
        edits (list): List of edits.

    Returns:
        list: List of edits with redundant edits removed.
    """
    indexes_to_remove = []
    i = 0
    while i < len(edits):
        if i+1 == len(edits):
            break
        first_line = edits[i].rstrip()  # Remove trailing whitespaces
        first_kind = first_line[0]
        first_line = first_line[2:]
        sec_line = edits[i + 1].rstrip()  # Remove trailing whitespaces
        sec_kind = sec_line[0]
        sec_line = sec_line[2:]
        if first_kind == "R" and sec_kind == "A" and first_line.lstrip() == sec_line.lstrip():
            if i + 2 < len(edits) and (edits[i + 2][:2] == "A:" or edits[i + 2][0] == "R"):
                indexes_to_remove.extend([i, i+1])
                i += 2
            elif i + 2 == len(edits):
                indexes_to_remove.extend([i, i+1])
                i += 2
            else:
                i += 1
        else:
            i += 1
    result_list = [edits[i] for i in range(len(edits)) if i not in set(indexes_to_remove)]
    return result_list



def marge_addition_edits(s):
    """
    Merges addition edits in the given list of edits.

    Args:
        s (list): List of edits.

    Returns:
        list: List of merged edits.
    """
    i = 0
    add_under = ""
    addition_lines = []
    indexs_to_remove = []
    while i < len(s):
        if s[i][:2] == "A:":
            add_under = s[i].split("<endRow>")[0][2:]
            indexs_to_remove.append(i)
            addition_lines.append(s[i].split("<endRow>")[1])
            while i + 1 < len(s) and s[i+1].split("<endRow>")[0] == "A:"+add_under:
                i += 1
                indexs_to_remove.append(i)
                addition_lines.append(s[i].split("<endRow>")[1])
            addition = "\n".join(addition_lines)
            s[indexs_to_remove[0]] = "A:"+add_under+"<endRow>"+addition
            # Delete indexes from s
            for index in sorted(indexs_to_remove[1:], reverse=True):
                del s[index]
            indexs_to_remove = []
            addition_lines = []
            add_under = ""
        i += 1
    
    
    return s
                




def get_edits(df):
    """
    Extracts edits from a DataFrame containing nonvul and vul columns.

    Args:
        df (pandas.DataFrame): DataFrame containing nonvul and vul columns.

    Returns:
        list: List of dictionaries, where each dictionary represents the edits for a sample.
    """
    all_edits = []
    for sample in range(len(df['nonvul'])):
        s = compare_functions(df['nonvul'].iloc[sample], df['vul'].iloc[sample])
        s = clear_edits(s)
        s = marge_addition_edits(s)
        edits = {}
        i = 0
        postfix = 'a'
        while i < len(s):
            if s[i][0] == "R":
                if i+1 == len(s) or (i+1 < len(s) and s[i+1] == "A EmptyLine"):
                    edits[s[i][1:]] = "EmptyLine"
                    i += 1
                elif i+1 < len(s) and s[i][1] != ":":
                    index, edits[s[i][1:]] = one_edit(s,i)
                    i = index
            else:
                key, value = s[i].split("<endRow>")
                if key not in edits.keys():
                    edits[key] = value
                else:
                    edits[key + postfix] = value
                    postfix = chr(ord(postfix) + 1)
            i += 1
        all_edits.append(edits)
    return all_edits



def get_inputs(df):
    """
    Get the 'nonvul' column values from the given DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame.

    Returns:
        list: A list of 'nonvul' column values.
    """
    inputs = []
    for i in range(len(df)):
        x = df['nonvul'].iloc[i]
        inputs.append(x)
    return inputs




def get_outpus(df, all_edits):
    """
    Generate outputs based on the given dataframe and all_edits dictionary.

    Args:
        df (pandas.DataFrame): The input dataframe.
        all_edits (list[dict]): A list of dictionaries containing the edits for each row in the dataframe.

    Returns:
        list[str]: A list of generated outputs.

    """
    outputs = []
    for i in range(len(df)):
        x = ""
        for key, value in all_edits[i].items():
            if key[-4:] != 'else' and 'a' <= key[-1] <= 'z':
                key = key[: -1]
            if not key[0] == "A":
                if value == "EmptyLine":
                    x += "<N>" + "\n" + key + "\n"
                    x += value + "\n"
                else:
                    x += "<N>" + "\n" + key + "\n"
                    x += value
            else:
                x += "<N>" + "\n" + key[2:] + "\n" + key[2:] +"\n" + value + "\n"
        x = x[:-1]
        outputs.append(x)
    return outputs



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
        try:
            nonvul = df['nonvul'].iloc[i]
            # nonvul = normalize_c_code(nonvul)
            nonvul = nonvul.replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")
            lines_after_fix = df['vul'].iloc[i]
            # lines_after_fix = normalize_c_code(lines_after_fix)
            lines_after_fix = lines_after_fix.replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")
            # Split the file content into functions (assuming functions are well-defined)
            row = nonvul + lines_after_fix  # Change this based on your function definitions
            function_hash = hashlib.sha256(row.encode()).hexdigest()
            if function_hash not in function_groups:
                function_groups[function_hash] = []
            function_groups[function_hash].append(i)
        except:
            continue

    indexes_to_drop = [index for index_list in function_groups.values() if len(index_list) > 1 for index in index_list[1:]]
    df = df.drop(indexes_to_drop, axis=0)
    df = df.reset_index(drop=True)
    return df






class CCodeNormalizer(c_ast.NodeVisitor):
    def __init__(self):
        self.variable_map = {}
        self.var_count = 1
    
    def visit_FuncDef(self, node):
        # Reset variable mapping for each function definition
        self.variable_map = {}
        self.var_count = 1
        self.generic_visit(node)
    
    def visit_Decl(self, node):
        # Only rename variables in declarations that are not function parameters or functions themselves
        if isinstance(node.type, c_ast.TypeDecl):
            if isinstance(node.type.type, c_ast.IdentifierType) and not isinstance(node.init, c_ast.FuncCall):
                identifier = node.name
                if identifier not in self.variable_map:
                    new_name = f"var{self.var_count}"
                    self.variable_map[identifier] = new_name
                    self.var_count += 1
                node.name = self.variable_map[identifier]
        self.generic_visit(node)
    
    def visit_ID(self, node):
        # Rename variable uses
        if node.name in self.variable_map:
            node.name = self.variable_map[node.name]
        self.generic_visit(node)

def normalize_c_code(code):
    parser = c_parser.CParser()
    ast = parser.parse(code)
    normalizer = CCodeNormalizer()
    normalizer.visit(ast)
    return ast

def ast_to_string(ast):
    """Convert AST back to code. This is a simple unparser."""
    return pycparser.c_generator.CGenerator().visit(ast)


# get train data
def get_train(path_trainset, full_vulgen=False):
    """
    Retrieves the training dataset for the injection model.

    Returns:
        train (pandas.DataFrame): The processed training dataset.
    """
    # get non-vul function
    data = pd.read_csv(path_trainset)
    data = append_spaces_suffix_to_duplicates(data, 'nonvul')
    vul_funcs = data[["vul"]]

    train = pd.DataFrame()
    train['nonvul'] = data['nonvul']
    train['vul'] = vul_funcs['vul']

    if full_vulgen:
        drop_index_path = 'pickle_files/location_with_spaces_to_delete_train.pkl'
        with open(drop_index_path, 'rb') as file:
            inject_indexes_to_delete = pickle.load(file)
        train = train.drop(inject_indexes_to_delete)
        train = train.reset_index(drop=True)
    train = drop_duplicates(train)
    print(len(train))
    return train




def get_test(dataset_path, full_vulgen=False):
    """
    Reads the test dataset from a CSV file, appends spaces suffix to duplicates in the 'nonvul' column,
    and performs data filtering based on pre-defined indexes stored in pickle files.
    
    Returns:
    test (pandas.DataFrame): The processed test dataset.
    """
    data = pd.read_csv(dataset_path)
    if 'cwe' not in data.columns:
        data['cwe'] = -1
    #get vul function
    vul_funcs = data[["vul"]]

    test = pd.DataFrame()
    test['nonvul'] = data['nonvul']
    test['vul'] = vul_funcs['vul']
    
    test['cwe'] = data['cwe']
    
    if full_vulgen:
        drop_index_path = 'pickle_files/location_with_spaces_to_delete_test.pkl'
        with open(drop_index_path, 'rb') as file:
            inject_indexes_to_delete = pickle.load(file)
        test = test.drop(inject_indexes_to_delete)
        test = test.reset_index(drop=True)
    test = drop_duplicates(test)
    test = test.reset_index(drop=True)
    print(len(test))
    return test



def get_test_eval():
    """
    Reads the test dataset from a CSV file, appends spaces suffix to duplicates in the 'nonvul' column,
    and performs data filtering based on pre-defined indexes stored in pickle files.
    
    Returns:
    test (pandas.DataFrame): The processed test dataset.
    """
    data = pd.read_csv(f"Dataset_VulGen/vulgen_test_with_diff_lines_spaces.csv")

    #get vul function
    vul_funcs = data[["vul"]]

    test = pd.DataFrame()
    test['nonvul'] = data['nonvul']
    test['vul'] = vul_funcs['vul']
    
    file_path = 'pickle_files/location_with_spaces_to_delete_test.pkl'
    with open(file_path, 'rb') as file:
        inject_indexes_to_delete = pickle.load(file)
    test = test.drop(inject_indexes_to_delete)
    test = test.reset_index(drop=True)

    print(len(test))
    return test




def change_pads_token_expect_the_first_one(labels, tokenizer_pad_token_id):
    """
    Modifies the labels tensor by replacing all occurrences of the pad token except the first one with -100.

    Args:
        labels (torch.Tensor): The labels tensor.
        tokenizer_pad_token_id (int): The ID of the pad token in the tokenizer.

    Returns:
        torch.Tensor: The modified labels tensor.
    """
    for idx, inner_tensor in enumerate(labels):
        if inner_tensor.numel() > 0:
            pad_indices = (inner_tensor == tokenizer_pad_token_id).nonzero(as_tuple=False)
            if pad_indices.numel() > 0:
                pad_index = pad_indices.min()
                inner_tensor[pad_index.item() + 1:] = -100
    return labels



def tokenize(data, tokenizer):
    """
    Tokenizes the input data using the provided tokenizer.

    Args:
        data (DataFrame): The input data containing 'inputs' and 'outputs' columns.
        tokenizer (Tokenizer): The tokenizer to be used for tokenization.

    Returns:
        dict: A dictionary containing the tokenized model inputs and labels.
    """
    model_inputs = tokenizer(data['inputs'].tolist(), return_tensors='pt', padding='max_length', truncation=True, max_length=2048)
    model_outputs = tokenizer(data['outputs'].tolist(), return_tensors='pt', padding='max_length', truncation=True, max_length=2048)
    labels = model_outputs['input_ids']
    pad_token_mask = labels == tokenizer.pad_token_id
    # labels[pad_token_mask] = -100

    # labels = change_pads_token_expect_the_first_one(labels, tokenizer.pad_token_id)
    model_inputs['labels'] = labels
    return model_inputs



def process_c_functions(df, column_name):
    for index, row in df.iterrows():
        lines = row[column_name].split('\n')  # Split the function into lines
        processed_lines = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line in ["{", "}"] and i > 0:  # Check if line contains only "{" or "}"
                # Append the character to the end of the previous line
                processed_lines[-1] = processed_lines[-1] + " " + line
            else:
                processed_lines.append(lines[i])  # Add the line as it is
            i += 1
        # Update the DataFrame with the processed function
        df.at[index, column_name] = '\n'.join(processed_lines)
    return df





def create_datasets(path_trainset, path_testset, full_vulgen=False):
    """
    Create tokenized datasets for training and testing.

    Args:
        tokenizer (Tokenizer): The tokenizer object to tokenize the input data.

    Returns:
        tokenized_train (Dataset): Tokenized training dataset.
        tokenized_test (Dataset): Tokenized testing dataset.
    """
    train = get_train(path_trainset, full_vulgen)
    # train = train[:1]
    # train = train[train['nonvul'].str.len() <= 4000]
    test = get_test(path_testset, full_vulgen)
    # test = test[:5]
    test_edits = get_edits(test)
    train_edits = get_edits(train)
    train['inputs'] = get_inputs(train)
    train['outputs'] = get_outpus(train, train_edits)
    test['inputs'] = get_inputs(test)
    test['outputs'] = get_outpus(test, test_edits)
    # test = test[test['inputs'].str.len() + test['outputs'].str.len() <= 1500]  # Drop rows where 'inputs' length is larger than 1500
    # train = train[:1]
    # test = test[:50]
    # train = Dataset.from_pandas(train)
    # test= Dataset.from_pandas(test)
    # tokenized_train = tokenize(train, tokenizer)
    # tokenized_test = tokenize(test, tokenizer)
    # plot_data_len(tokenized_train, tokenized_test)
    # tokenized_train = Dataset.from_dict(tokenized_train)
    # tokenized_test = Dataset.from_dict(tokenized_test)
    
    return train, test



def plot_data_len(tokenized_train, tokenized_test):
    lengths = [len(x) for x in tokenized_train['input_ids']]
    lengths += [len(x) for x in tokenized_test['input_ids']]
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=20, alpha=0.7, color='blue')
    plt.xlabel('Length of the input sequence')
    plt.ylabel('Frequency')
    plt.title('Distribution of input sequence lengths')
    plt.show()
    print("Max length of input sequence:", max(lengths))
               
            

def get_testset_for_eval(tokenizer, path_testset, all_vulgen, drop_index_path):
    test = get_test(path_testset, full_vulgen=all_vulgen)
    test_edits = get_edits(test)
    test['inputs'] = get_inputs(test)
    test['outputs'] = get_outpus(test, test_edits)
    tokenized_test = tokenize(test, tokenizer)
    tokenized_test = Dataset.from_dict(tokenized_test)
    return tokenized_test



def get_testset_for_replace(dataset_path, all_vulgen):
    test = get_test(dataset_path, all_vulgen)
    test_edits = get_edits(test)
    test['inputs'] = get_inputs(test)
    test['outputs'] = get_outpus(test, test_edits)
    return test

def too_large_input(tokenized_train, tokenized_test):
    """
    Check if the input tensors in the tokenized_train and tokenized_test dictionaries are too large.
    
    Args:
        tokenized_train (dict): Dictionary containing tokenized training data.
        tokenized_test (dict): Dictionary containing tokenized test data.
    
    Returns:
        tuple: A tuple containing two lists. The first list contains the indices of the training data tensors
               that are too large, and the second list contains the indices of the test data tensors that are too large.
    """
    too_long_to_delete_train = []
    too_long_to_delete_test = []
    for idx, tensor in enumerate(tokenized_train['input_ids']):
        if tensor[2047] != 50256:
            too_long_to_delete_train.append(idx)
    for idx, tensor in enumerate(tokenized_test['input_ids']):
        if tensor[2047] != 50256:
            too_long_to_delete_test.append(idx)
    return too_long_to_delete_train, too_long_to_delete_test

# train, test = create_datasets(AutoTokenizer.from_pretrained('Salesforce/codet5p-6b'), "Datasets/vulgen_train_with_diff_lines_spaces.csv", "Datasets/vulgen_test_with_diff_lines_spaces.csv", True)
# too_long_to_delete_train, too_long_to_delete_test = too_large_input(train, test)
# with open('pickle_files/inject_with_spaces_to_delete_train.pkl', 'wb') as file:
#     pickle.dump(too_long_to_delete_train, file)
# with open('pickle_files/inject_with_spaces_to_delete_test.pkl', 'wb') as file:
#     pickle.dump(too_long_to_delete_test, file)

