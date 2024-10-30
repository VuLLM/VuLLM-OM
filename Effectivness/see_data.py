import pandas as pd
import hashlib
import re
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



def drop_df1_duplicates_in_df2(df1, df2):
    indexes = []
    function_df1 = {}
    function_df2 = {}
    count = 0
    for i in range(len(df1)):
        try:
            vul = df1['func'].iloc[i]
            vul = vul.replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")
            # Split the file content into functions (assuming functions are well-defined)
            function_hash = hashlib.sha256(vul.encode()).hexdigest()
            if function_hash not in function_df1:
                function_df1[function_hash] = []
            function_df1[function_hash].append(i)
        except:
            continue
    for i in range(len(df2)):
        try:
            vul = df2['func'].iloc[i]
            # nonvul = Prepare_dataset_with_only_replace_only_encoder.normalize_c_code(nonvul)
            vul = vul.replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")
            # Split the file content into functions (assuming functions are well-defined)
            function_hash = hashlib.sha256(vul.encode()).hexdigest()
            if function_hash not in function_df2:
                function_df2[function_hash] = []
            function_df2[function_hash].append(i)
        except:
            continue
    for key in function_df1.keys():
        if key in function_df2.keys():
            indexes.append((df1.func[function_df1[key]], df2.func[function_df2[key]]))
            df1 = df1.drop(function_df1[key], axis=0)
            count += len(function_df1[key])
    df1 = df1.reset_index(drop=True)
    print(count)
    return df1




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


def remove_comments_and_empty_lines(code):
    # Remove single-line comments
    code = re.sub(r'//.*', '', code)
    # Remove multi-line comments
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    # Split the code into lines, remove completely empty lines, and join back
    lines = code.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    return '\n'.join(non_empty_lines)


def stratified_sample(df, target_column, total_samples=2000, ratio_1_to_0=1/4):
    # Calculate the number of samples for each target
    n_target_0 = int(total_samples / (ratio_1_to_0 + 1))
    n_target_1 = total_samples - n_target_0

    # Split the DataFrame into two based on the target
    df_target_1 = df[df[target_column] == 1]
    df_target_0 = df[df[target_column] == 0]

    # Sample from each group
    sampled_target_1 = df_target_1.sample(n=n_target_1, random_state=42)
    sampled_target_0 = df_target_0.sample(n=n_target_0, random_state=42)

    # Combine the samples
    sampled_df = pd.concat([sampled_target_1, sampled_target_0])

    # Shuffle the combined DataFrame
    sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)

    return sampled_df


if __name__ == '__main__':
    add_vuls = pd.read_csv('Effectivness/VuLLM_genearted_vuls/shorter_than20_create_newVul_5000.csv')
    add_vuls = add_vuls.sample(1000)
    print(len(add_vuls))
    # add_vuls.rename(columns={'vul': 'func'}, inplace=True)
    add_vuls.dropna(subset=['func'], inplace=True)
    add_vuls['target'] = 1
    print(len(add_vuls))

    # print(len(df2))
    # df2['target'] = 1
    add_nonvul = pd.read_csv('Effectivness/train_linevul/imbalanced/33600_samples_linevul_nonvul_addition.csv')
    # vgx = pd.read_csv('Effectivness/VGX_generated_15097.csv')
    # syn = syn.rename(columns={'label': 'target'})
    add_nonvul = add_nonvul.sample(4000)
    bigvul = pd.read_csv('Effectivness/train_linevul/imbalanced/train_plus_val_big-vul_20000.csv')
    bigvul = bigvul.rename(columns={'processed_func': 'func'})
    bigvul.drop(columns=['Unnamed: 0'], inplace=True)
    concatenated_df = pd.concat([add_vuls, add_nonvul, bigvul])
    print(concatenated_df.target.value_counts())
    print(concatenated_df.shape)
    concatenated_df = drop_duplicates(concatenated_df)
    print(concatenated_df.shape)
    test = pd.read_csv('detector_models/LineVul/code/data/train_linevul_test_reveal/test.csv')
    concatenated_df = drop_df1_duplicates_in_df2(concatenated_df, test)
    print(concatenated_df.shape)
    concatenated_df.dropna(subset=['func', 'target'], inplace=True)
    concatenated_df['func'] = concatenated_df['func'].apply(remove_comments_and_empty_lines)
    concatenated_df.reset_index(drop=True, inplace=True)
    concatenated_df = concatenated_df.sample(frac=1).reset_index(drop=True)
    print(concatenated_df.target.value_counts())

    concatenated_df.to_csv('detector_models/LineVul/code/data/train_linevul_test_reveal/VuLLM_OM_add_1000_vuls.csv', index=False)
    
    
    # df1 = pd.read_csv('detector_models/LineVul/code/data/train_linvul_balanced_test_primevul/linevul_train_primevul_balanced.csv')
    # df2 = pd.read_csv('detector_models/LineVul/code/data/train_linvul_balanced_test_primevul/linevul_test_primevul_balanced.csv')
    
    # df1 = drop_df1_duplicates_in_df2(df1, df2)
    # df1.to_csv('detector_models/LineVul/code/data/train_linvul_balanced_test_primevul/linevul_train_primevul_balanced.csv', index=False)
    

    
    
    
    # df = pd.read_csv('Effectivness/VuLLM_genearted_vuls/vullm_479_vulgen_after_gen_all_data_train.csv')
    # print(df.shape)
    # df = df[~df['vul'].isin(["Line not exist in the function", "No change in the function"])]
    # print(df.shape)
    # # df = df[df['vul'].str.len() <= 715]
    # print(df.shape)
    # df.dropna(subset=['vul'], inplace=True)
    # df.rename(columns={'vul': 'func'}, inplace=True)
    # df['target'] = 1
    # print(len(df))
    # df.to_csv('Effectivness/VuLLM_genearted_vuls/vullm_479_vulgen_after_gen_all_data_train.csv', index=False)
    
    # df = pd.read_csv('detector_models/LineVul/code/data/train_linevul_test_reveal/train_plus_val_big-vul.csv')
    # print(df.shape)
    # print(df.target.value_counts())
 
    


    # df = pd.read_csv('Effectivness/train_linevul/linevul_nonvul_addition.csv')

    # # target_proportions = df['target'].value_counts(normalize=True)

    # # Perform stratified sampling to get 36,600 samples
    # sampled_df, _ = train_test_split(df, 
    #                                 train_size=33600, 
    #                                 stratify=df['target'], 
    #                                 random_state=42)

    # sampled_df.to_csv('Effectivness/train_linevul/33600_samples_linevul_nonvul_addition.csv', index=False)
    
    
    # df = pd.read_csv('Effectivness/VGX_generated_15097.csv')
    # print(df.shape)
    # print(df.columns)

    # # Calculate the length of each function
    # df['func_length'] = df['func'].str.len()
    # plt.hist(df['func_length'], bins=20, range=(0, 2000))
    # plt.xlabel('Function Length')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Function Length')
    # plt.savefig('function_length_histogram.png')

    # big_vul = pd.read_csv('Effectivness/train_linevul/imbalanced/linevul_nonvul_addition_1852.csv')
    # print(big_vul.target.value_counts())
    # sampled_df, _ = train_test_split(big_vul, train_size=1852, 
    #                                  stratify=big_vul['target'], random_state=42)
    # print(sampled_df.target.value_counts())
    # sampled_df.to_csv('Effectivness/train_linevul/linevul_nonvul_addition_1852.csv', index=False)
    # sampled_df = stratified_sample(big_vul, target_column='target', total_samples=20000, ratio_1_to_0=1/4)
    # print(sampled_df.target.value_counts())
    # sampled_df = big_vul.sample(542)
    # sampled_df.to_csv('Effectivness/train_linevul/balanced/linevul_nonvul_addition_542.csv.csv', index=False)
    
    # train = pd.read_csv('Datasets/primevul_pairs.csv')
    # test = pd.read_csv('detector_models/LineVul/code/data/train_linvul_balanced_test_primevul/devign_test_linevul_balanced.csv')
    # train['func'] = train['vul'].apply(remove_comments_and_empty_lines)
    # train['func2'] = train['nonvul'].apply(remove_comments_and_empty_lines)
    # new_col = pd.concat([train['func'], train['func2']], ignore_index=True)
    # train['func'] = new_col
    # test['func'] = test['func'].apply(remove_comments_and_empty_lines)
    # # print(train.columns)
    # print(train.shape)
    # print(test.columns)
    # print(test.shape)
    # test = drop_df1_duplicates_in_df2(test, train)
    # print(test.shape)
    # print(test.target.value_counts())
    # df_target_1 = test[test['target'] == 1]
    # df_target_0 = test[test['target'] == 0].sample(n=len(df_target_1))
    # df = pd.concat([df_target_1, df_target_0])
    # print(df.target.value_counts())
    # print(df.columns)
    # df.to_csv('detector_models/LineVul/code/data/train_linvul_balanced_test_primevul/devign_test_linevul_balanced.csv', index=False)

    # train = pd.read_csv('Datasets/vulgen_datasets/vulgen_train_drop_dup.csv')
    # test = pd.read_csv('Datasets/vulgen_datasets/vulgen_test_775.csv')
    # print(train.shape)
    # train = drop_df1_duplicates_in_df2(train, test)
    # print(train.shape)