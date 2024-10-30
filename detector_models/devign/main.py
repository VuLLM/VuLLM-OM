# -*- coding: utf-8 -*-
"""
    This module is intended to join all the pipeline in separated tasks
    to be executed individually or in a flow by using command-line options

    Example:
    Dataset embedding and processing:
        $ python taskflows.py -e -pS
"""
import os
import argparse
import gc
import shutil
from argparse import ArgumentParser
import torch
from gensim.models.word2vec import Word2Vec
import pandas as pd
import configs
from tqdm import tqdm
from gensim.models import Word2Vec
import src.data as data
import src.prepare as prepare
import src.process as process
import src.utils.functions.cpg as cpg
import numpy as np
import random
PATHS = configs.Paths()
FILES = configs.Files()
DEVICE = FILES.get_device()


def select(dataset):
    sorted_list = sorted(dataset['code'], key=len, reverse=True)
    length_list = [len(string) for string in sorted_list]
    len_filter = dataset.code.str.len() < 5000
    result = dataset[len_filter]
    print(result['target'].value_counts())
    print(f"Count of dataset.code.str.len() < 5000: {len(result)}")
    return result


def create_task():
    context = configs.Create()
    raw = data.read(PATHS.raw, FILES.raw)
    filtered = data.apply_filter(raw, select)
    filtered = data.clean(filtered)
    print(len(filtered))
    # data.drop(filtered, ["file_name"])
    slices = data.slice_frame(filtered, context.slice_size)
    slices = [(s, slice.apply(lambda x: x)) for s, slice in slices]

    cpg_files = []
    # Create CPG binary files
    for s, slice in slices:
        data.to_files(slice, PATHS.joern)
        cpg_file = prepare.joern_parse(context.joern_cli_dir, PATHS.joern, PATHS.cpg, f"{s}_{FILES.cpg}")
        cpg_files.append(cpg_file)
        print(f"Dataset {s} to cpg.")
        shutil.rmtree(PATHS.joern)
    # Create CPG with graphs json files
    # cpg_files = [f"{s}_{FILES.cpg}.bin" for s in range(len(slices))][43:]
    json_files = prepare.joern_create(context.joern_cli_dir, PATHS.cpg, PATHS.cpg, cpg_files)
    # json_files = [f"{s}_{FILES.cpg}.json" for s in range(len(slices))][43:]
    # files = json_files
    # json_files = json_files[:42]
    # json_files.extend(files[43:])
    for (s, slice), json_file in zip(slices, json_files):
        graphs = prepare.json_process(PATHS.cpg, json_file)
        if graphs is None:
            print(f"Dataset chunk {s} not processed.")
            continue
        dataset = data.create_with_index(graphs, ["Index", "cpg"])
        dataset = data.inner_join_by_index(slice, dataset)
        print(f"Writing cpg dataset chunk {s}.")
        data.write(dataset, PATHS.cpg, f"{s}_{FILES.cpg}.pkl")
        del dataset
        gc.collect()


def embed_task():
    context = configs.Embed()
    # Tokenize source code into tokens
    dataset_files = data.get_directory_files(PATHS.cpg)
    # w2vmodel = Word2Vec(**context.w2v_args)
    w2v_init = False

    w2vmodel = Word2Vec.load('data/w2v/w2v.model')
    cpg_dataset = pd.DataFrame()
    # i = 0
    # for pkl_file in dataset_files:
    #     print(i)
    #     file_name = pkl_file.split(".")[0]
    #     small_dataset = data.load(PATHS.cpg, pkl_file)
    #     cpg_dataset = pd.concat([cpg_dataset, small_dataset], ignore_index=True)
    #     i += 1
    # tokens_dataset = data.tokenize(cpg_dataset)
    # w2vmodel.build_vocab(sentences=tokens_dataset.tokens, update=not w2v_init)
    # w2vmodel.train(tokens_dataset.tokens, total_examples=w2vmodel.corpus_count, epochs=50)
    # print("Saving w2vmodel.")
    # w2vmodel.save(f"{PATHS.w2v}/{FILES.w2v}")

    for pkl_file in dataset_files:
        if os.path.exists(os.path.join("data/input/", pkl_file.split(".")[0] + "_input.pkl")):
            continue
        file_name = pkl_file.split(".")[0]
        cpg_dataset = data.load(PATHS.cpg, pkl_file)
        tokens_dataset = data.tokenize(cpg_dataset)
        data.write(tokens_dataset, PATHS.tokens, f"{file_name}_{FILES.tokens}")
        # word2vec used to learn the initial embedding of each token
        # w2vmodel.build_vocab(sentences=tokens_dataset.tokens, update=not w2v_init)
        # w2vmodel.train(tokens_dataset.tokens, total_examples=w2vmodel.corpus_count, epochs=1)
        # if w2v_init:
        #     w2v_init = False
        # Embed cpg to node representation and pass to graph data structure
        cpg_dataset["nodes"] = cpg_dataset.apply(lambda row: cpg.parse_to_nodes(row.cpg, context.nodes_dim), axis=1)
        # remove rows with no nodes
        cpg_dataset = cpg_dataset.loc[cpg_dataset.nodes.map(len) > 0]
        inputs = []
        index_to_drop = []
        for index, row in cpg_dataset.iterrows():
            try:
                inputs.append(prepare.nodes_to_input(row.nodes, row.target, context.nodes_dim,
                                                w2vmodel.wv, context.edge_type))
            except:
                index_to_drop.append(index)
                inputs.append(None)
        data.drop(cpg_dataset, ["nodes"])
        cpg_dataset["input"] = inputs
        cpg_dataset.drop(index_to_drop, inplace=True)
        print(f"Saving input dataset {file_name} with size {len(cpg_dataset)}.")
        data.write(cpg_dataset[["input", "target", "test"]], PATHS.input, f"{file_name}_{FILES.input}")
        del cpg_dataset
        gc.collect()



def process_task(stopping, num_epochs):
    context = configs.Process()
    context.epochs = num_epochs
    devign = configs.Devign()
    model_path = PATHS.model + FILES.model
    print("Start loading model")
    model = process.Devign(path=model_path, device=DEVICE, model=devign.model, learning_rate=devign.learning_rate,
                           weight_decay=devign.weight_decay,
                           loss_lambda=devign.loss_lambda)
    train = process.Train(model, context.epochs)
    input_dataset = data.loads(PATHS.input)
    train_loader, test_loader = list(
        map(lambda x: x.get_loader(context.batch_size, shuffle=context.shuffle),
            data.train_val_test_split(input_dataset, shuffle=context.shuffle)))

    train_loader_step = process.LoaderStep("Train", train_loader, DEVICE)
    test_loader_step = process.LoaderStep("Test", test_loader, DEVICE)

    if stopping:
        early_stopping = process.EarlyStopping(model, patience=context.patience)
        train(train_loader_step, test_loader_step, early_stopping)
        model.load()
    else:
        train(train_loader_step, test_loader_step)
        model.save()

    process.predict(model, test_loader_step)


def main():
    """
    main function that executes tasks based on command-line options
    """
    # create_task()
    parser: ArgumentParser = argparse.ArgumentParser()
    # parser.add_argument('-p', '--prepare', help='Prepare task', required=False)
    parser.add_argument('-c', '--create', action='store_true')
    parser.add_argument('-e', '--embed', action='store_true')
    parser.add_argument('-p', '--process', action='store_true')
    parser.add_argument('-pS', '--process_stopping', action='store_true')
    parser.add_argument('-n', '--num_epochs', type=int, help='Number of epochs')

    args = parser.parse_args()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed = 333
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Ensures reproducibility of hash-based operations in Python
    os.environ['PYTHONHASHSEED'] = str(seed)

    if args.create:
        create_task()
    if args.embed:
        embed_task()
    if args.process:
        process_task(False, args.num_epochs)
    if args.process_stopping:
        process_task(True, args.num_epochs)



if __name__ == "__main__":
    main()
