import pandas as pd
import json
import random

# df = pd.read_csv('data/raw/shorter_than20_create_newVul_460.csv')
# print(len(df))
# df = df[~df['func'].isin(["Line not exist in the function", "No change in the function"])]
# print(len(df))
# df.dropna(subset=['func'], inplace=True)
# print(len(df))
# print(df.columns)
# print(df['target'].value_counts())
# print(len(df))

# # Create a list to store the JSON objects
# json_data = []

# # Iterate through the DataFrame rows
# for index, row in df.iterrows():
#     json_object = {
#         "code": row['func'],
#         "target": int(row['target']),  # Assuming 'target' is already 0 or 1
#         "test": 0
#     }
#     json_data.append(json_object)

# # Convert the list of dictionaries to JSON format
# json_output = json.dumps(json_data, indent=2)

# # If you want to save this to a file:
# with open('data/raw/vullm_wild_460.json', 'w') as f:
#     f.write(json_output)

# If you want to print it to console:
# print(json_output)

# # If you need it as a list of dictionaries for further processing:
# # json_data is already in that format

# # If you need it as a JSON string without indentation:
# json_string = json.dumps(json_data)

# Read the JSON file
with open('detector_models/devign/data/raw/vullm_omne_model_gen_data_792.json', 'r') as f:
    json_data = json.load(f)

# Print the JSON data

# Shuffle the JSON data randomly
random.shuffle(json_data)

# Select the first 470 records
selected_data = json_data[:470]

# Print the selected JSON data
# Write the selected_data to the JSON file
with open('detector_models/devign/data/raw/vullm_omne_model_gen_data_792.json', 'w') as f:
    json.dump(selected_data, f)