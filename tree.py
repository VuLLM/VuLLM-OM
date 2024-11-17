import os

# List of folders and files to exclude from the tree output
EXCLUDED_ITEMS = {
    ".neptune", ".vscode", "Charts", "code_files/__pycache__", "Datasets",
    "detector_models/devign", "detector_models/LineVul", "Effectivness", 
    "gen_vuls", "pickle_files", "saved_models", "cwe_hist_tm.png", 
    "cwe_hist.pickle", "cwe_hist.png", "cwe.pickle", "tree.py", 
    "vgx_479_nonvul.csv", "utils/__pycache__", ".git"
}

def generate_tree(directory, prefix=""):
    tree = []
    contents = sorted(os.listdir(directory))
    for index, item in enumerate(contents):
        path = os.path.join(directory, item)
        
        # Check if the item is in the exclusion list
        if item in EXCLUDED_ITEMS or any(excluded in path for excluded in EXCLUDED_ITEMS):
            continue

        connector = "└── " if index == len(contents) - 1 else "├── "
        
        if os.path.isdir(path):
            tree.append(f"{prefix}{connector}{item}/")
            # Recursively generate the tree for subdirectories
            tree.extend(generate_tree(path, prefix + ("    " if index == len(contents) - 1 else "│   ")))
        else:
            tree.append(f"{prefix}{connector}{item}")

    return tree

def print_tree(directory):
    print(f"{os.path.basename(directory)}/")
    for line in generate_tree(directory):
        print(line)

# Replace 'your_project_directory' with the path to your project directory
project_directory = "/home/urizlo/VuLLM_One_Stage"
print_tree(project_directory)
