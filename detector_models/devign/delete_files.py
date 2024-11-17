import os

directory = 'data/vullm_input'

for filename in os.listdir(directory):
    if not filename.endswith('input.pkl'):
        file_path = os.path.join(directory, filename)
        os.remove(file_path)