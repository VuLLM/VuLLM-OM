import os
number = 800
folder_path = "data/vullm_one_model/"
name = 'vullm_one_model'

for i in range(0,3):
    if not os.path.exists(os.path.join(folder_path, f"{i}_cpg.json")):
        continue
    old_file_name = os.path.join(folder_path, f"{i}_cpg.bin")
    new_file_name = os.path.join(folder_path, f"{name}{i}_cpg.bin")
    os.rename(old_file_name, new_file_name)

    old_file_name = os.path.join(folder_path, f"{i}_cpg.pkl")
    new_file_name = os.path.join(folder_path, f"{name}{i}_cpg.pkl")
    os.rename(old_file_name, new_file_name)

    old_file_name = os.path.join(folder_path, f"{i}_cpg.json")
    new_file_name = os.path.join(folder_path, f"{name}{i}_cpg.json")
    os.rename(old_file_name, new_file_name)
