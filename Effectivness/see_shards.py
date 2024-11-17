import ast
import pandas as pd

def read_dicts_from_file(file_path):
    df = pd.DataFrame()
    try:
        with open(file_path, 'r') as file:
            file.read(1)  # Read and ignore the opening bracket '['
            funcs, labels =  process_file(file)
            df['func'] = funcs
            df['label'] = labels
            return df
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")

def process_file(file):
    funcs = []
    labels = []
    partial_data = ''
    dict_count = 0
    while True:
        char = file.read(1)
        if not char:
            break  # End of file

        partial_data += char

        # Check for the specific ending patterns
        if partial_data.endswith('''\"label\": 1}''') or partial_data.endswith('''\"label\": 0}'''):
            # Process the dictionary once the end pattern is detected
            try:
                # Trim the last character (the closing bracket of the list) if it's the end
                if partial_data.endswith(','):
                    partial_data = partial_data[:-1]
                
                dict_obj = ast.literal_eval(partial_data)
                funcs.append(dict_obj['code'])
                labels.append(dict_obj['label'])
                dict_count += 1
                print(dict_count)
                partial_data = ''  # Reset for the next dictionary
            except SyntaxError as e:
                print(f"Error parsing dictionary: {e}")

            # Prepare to read the next part
            next_char = file.read(1)
            if next_char and next_char not in ', \n':
                partial_data += next_char  # Start the new dictionary immediately if no comma or newline

    return funcs, labels

if __name__ == '__main__':
    # full_df = pd.DataFrame()
    # for i in range(1, 5):
    #     file_path = f'syn15029.json/syn15029.json.shard{i}'
    #     df = read_dicts_from_file(file_path)
    #     full_df = pd.concat([full_df, df], ignore_index=True)
    # full_df.to_csv('syn15029.csv', index=False)
    
    df = pd.read_csv('Effectivness/train_devign/train_devign.csv')
    print(df.columns)
    print(df.shape)
    print(df.target.value_counts())

