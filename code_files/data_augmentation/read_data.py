import pandas as pd
import aug
import sys
sys.path.append('/sise/home/urizlo/VuLLM_One_Stage/code_files')
from preprocess_data import Prepare_dataset_with_only_replace_only_encoder
import pickle
# Specify the file path
train = Prepare_dataset_with_only_replace_only_encoder.get_train("Datasets/vulgen_train_with_diff_lines_spaces.csv", True)
# Read the pickle file
with open('/home/urizlo/VuLLM_One_Stage/pickle_files/indexes_to_aug.pkl', 'rb') as f:
    indexes_to_aug = pickle.load(f)

aug_df = train.loc[indexes_to_aug]
nonvul_generated_functions = []
vul_generated_functions = []
for i in range(len(aug_df)):
    try:
        nonvul_generated_functions.append(aug.replace_for_with_while(aug_df.iloc[i]['nonvul']))
        vul_generated_functions.append(aug.replace_for_with_while(aug_df.iloc[i]['vul']))
    except:
        print(i)
    
    # Create a DataFrame with the two lists
generted_df = pd.DataFrame({'nonvul': nonvul_generated_functions, 'vul': vul_generated_functions})
x = 3


        
        
