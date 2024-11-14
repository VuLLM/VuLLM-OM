import hashlib
import pandas as pd
import json

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



def drop_df1_duplicates_in_df2(df1, df2):
    function_df1 = {}
    function_df2 = {}
    count = 0
    for i in range(len(df1)):
        try:
            nonvul = df1['nonvul'].iloc[i]
            nonvul = nonvul.replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")
            # nonvul = Prepare_dataset_with_only_replace_only_encoder.normalize_c_code(nonvul)
            lines_after_fix = df1['vul'].iloc[i]
            # lines_after_fix = Prepare_dataset_with_only_replace_only_encoder.normalize_c_code(lines_after_fix)
            lines_after_fix = lines_after_fix.replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")
            # Split the file content into functions (assuming functions are well-defined)
            row = nonvul + lines_after_fix  # Change this based on your function definitions
            function_hash = hashlib.sha256(row.encode()).hexdigest()
            if function_hash not in function_df1:
                function_df1[function_hash] = []
            function_df1[function_hash].append(i)
        except:
            continue
    for i in range(len(df2)):
        try:
            nonvul = df2['nonvul'].iloc[i]
            # nonvul = Prepare_dataset_with_only_replace_only_encoder.normalize_c_code(nonvul)
            nonvul = nonvul.replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")
            lines_after_fix = df2['vul'].iloc[i]
            # lines_after_fix = Prepare_dataset_with_only_replace_only_encoder.normalize_c_code(lines_after_fix)

            lines_after_fix = lines_after_fix.replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")
            # Split the file content into functions (assuming functions are well-defined)
            row = nonvul + lines_after_fix  # Change this based on your function definitions
            function_hash = hashlib.sha256(row.encode()).hexdigest()
            if function_hash not in function_df2:
                function_df2[function_hash] = []
            function_df2[function_hash].append(i)
        except:
            continue
    for key in function_df1.keys():
        if key in function_df2.keys():
            df1 = df1.drop(function_df1[key], axis=0)
            count += len(function_df1[key])
    df1 = df1.reset_index(drop=True)
    print(count)
    return df1



if __name__ == '__main__':
    # df = pd.read_csv('Effectivness/VuLLM_genearted_vuls/shorter_than20_create_newVul.csv')
    # print(len(df))
    # df = drop_duplicates(df)
    # print(len(df))
    # df = df[(df['func'].str.len() <= 715) & (df['func'].str.len() > 250)]
    # print(len(df))
    # df = df[~df['func'].isin(["Line not exist in the function", "No change in the function"])]
    # df['name'] = df['func'].apply(lambda x: x.split('(')[0].split()[-1])
    # names = df['name'].tolist()
    # df.drop_duplicates(subset=['name'], inplace=True, keep='first')
    # print(len(df))
    # df = df.sample(n=5000)
    # df.to_csv('Effectivness/VuLLM_genearted_vuls/shorter_than20_create_newVul_5000.csv', index=False)
    
    # data = []
    # with open('Effectivness/datasets/primevul_train.jsonl', 'r') as file:
    #     for line in file:
    #         data.append(json.loads(line))
    # df = pd.DataFrame(data)
    # print(len(df))
    # df = df[['func', 'target']]
    # df_target_1 = df[df['target'] == 1]
    # df_target_0 = df[df['target'] == 0].sample(n=len(df_target_1))
    # df = pd.concat([df_target_1, df_target_0])
    # print(df.target.value_counts()) 
    # df.to_csv('detector_models/LineVul/code/data/train_linvul_balanced_test_msr/devign_test_linevul_balanced.csv')
    # df = pd.read_csv('Effectivness/VuLLM_genearted_vuls/shorter_than20_create_newVul_5000.csv')
    # df['name'] = df['func'].apply(lambda x: x.split('(')[0].split()[-1])
    # names = df['name'].tolist()
    # div = pd.read_csv('Effectivness/datasets/diversevul.csv')
    # print(len(div))
    # div = div[div['target'] != 1]
    # div.dropna(subset=['func'], inplace=True)
    # print(len(div))
    # def extract_name(x):
    #     try:
    #         return x.split('(')[0].split()[-1]
    #     except IndexError:
    #         return None

    # div['name'] = div['func'].apply(extract_name)
    # div = div[div['name'].isin(names)]
    # div.drop_duplicates(subset=['name'], inplace=True, keep='first')
    # print(len(div))
    # div.to_csv('Effectivness/datasets/diversevul_TM_for_gen.csv', index=False)
    
    df = pd.read_csv('Effectivness/datasets/diversevul_inRange_250_1000_16000Samples.csv')
    # print(df.columns)
    # print(df.shape)
    # print(df.target.value_counts())
    df.rename(columns={'func': 'nonvul'}, inplace=True)
    df_sample = df.sample(n=1050)
    df_sample.to_csv('Effectivness/datasets/divers_1050_nonvuls.csv', index=False)
