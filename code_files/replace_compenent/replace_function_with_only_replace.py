from uu import Error
import pandas as pd
import pickle
import sys
sys.path.append('/sise/home/urizlo/VuLLM_One_Stage')
from code_files import Prepare_dataset_with_only_replace
from collections import defaultdict
import hashlib
from dotenv import load_dotenv
import argparse

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
        lines_after_fix = df['vul'].iloc[i]
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



def get_data(path_testset, path_res_mod, all_vulgen):
    test = pd.read_csv(path_testset)
    test = process_c_functions(test, 'nonvul') # maybe already done in the csv --- need to check
    if all_vulgen:
        file_path = 'pickle_files/location_with_spaces_to_delete_test.pkl'
        with open(file_path, 'rb') as file:
            too_long_to_delete = pickle.load(file)
        test = test.drop(too_long_to_delete)
        test = test.reset_index(drop=True)
    test = drop_duplicates(test)
    test = test.drop(index=551).reset_index(drop=True)
    nonvul = []
    real_modifications = []
    inject_turth = Prepare_dataset_with_only_replace.get_testset_for_replace(path_testset, all_vulgen)
    for i in range(len(inject_turth)):
        nonvul.append(inject_turth['inputs'].iloc[i])
        real_modifications.append(inject_turth['outputs'].iloc[i])
    res_mod = None
    
    # get vul function
    vul_funcs = test[["vul"]]

    return nonvul, vul_funcs, res_mod, real_modifications
    # return nonvul, locations, real_inj



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
    



def get_new_vuls(nonvul, res_mod):
    """
    Get new vulnerabilities by injecting vulnerabilities into non-vulnerable locations.

    Args:
        nonvul (list): List of non-vulnerable functions.
        locations (list): List of locations to inject vulnerabilities.
        real_inj (list): List of real injection locations.

    Returns:
        list: List of new vulnerabilities.

    Raises:
        Exception: If an error occurs during vulnerability injection.
    """
    vuls = []
    num_of_problem = 0
    for i in range(len(nonvul)):
        mod_lines = res_mod[i].split('\n')
        locations, injections = get_locations_and_injections(mod_lines)
        if not (contain(locations, nonvul[i])) and locations != "Empty":
            print("Location not present in non-vulnerable list: ", i)
            vuls.append("no location")
            num_of_problem += 1
            continue
        flag = True
        if len(locations) == 0:
            flag = False
        for loc, inj in zip(locations, injections):
            if inj != loc:
                flag = False
                break
        if flag:
            print("Injection model could not create vulnerability in this location: ", i)
            vuls.append("wrong location")
            num_of_problem += 1
        else:
            try:
                vul = inject_vul(nonvul[i], locations, injections)
            except Exception as e:
                print(f"Error occurred at index {i}: {str(e)}")
                vuls.append(str(e)) 
                num_of_problem += 1
                continue
            vuls.append(vul)
    return vuls, num_of_problem



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
        x = delete_pre_spaces(vuls[i])
        y = delete_pre_spaces(vul_funcs['vul'].iloc[i]) 
        if x == y:
            count += 1
        else:
            not_good.append(i)
    print(count/len(vuls))
    return not_good


def main(path_testset, path_res_mod, all_vulgen, output_dir):
    load_dotenv()
    nonvul, vul_funcs, res_mod, real_modifications = get_data(path_testset, path_res_mod, all_vulgen)
    vuls, num_of_problem = get_new_vuls(nonvul, real_modifications)
    print(num_of_problem)
    not_good = get_accurecy(vuls, vul_funcs)
    print(not_good)
    df = pd.DataFrame()
    df['non-vul'] = nonvul
    df['vul'] = vuls
    df.to_csv(output_dir, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model with specific command line arguments.')
    parser.add_argument('--path_testset', type=str, default='Dataset_VulGen/vulgen_test_775_with_diff_lines_spaces.csv', help='Path to test set csv file')
    parser.add_argument('--path_res_mod', type=str, default='res_mod/vulgen_res_mod.csv', help='Path to localization results of the testset')
    parser.add_argument('--all_vulgen', type=bool, default=False, help='Is the test set is the whole test set of VulGen?')
    parser.add_argument('--output_dir', type=str, default='connected_models/generated_vul/vulgen_res.csv', help='Path to where to save the new vulnerable functions csv file')
    args = parser.parse_args()
    main(args.path_testset, args.path_res_mod, args.all_vulgen, args.output_dir)
