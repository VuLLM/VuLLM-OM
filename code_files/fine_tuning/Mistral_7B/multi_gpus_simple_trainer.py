from transformers import TrainingArguments, DataCollatorForLanguageModeling, Trainer, Seq2SeqTrainingArguments
import torch
import numpy as np
import neptune
# from accelerate import Accelerator
import evaluate
import os
# import ..utils
import sys
sys.path.append('/sise/home/urizlo/VuLLM_One_Stage')
from utils import Mistral_7B, Create_lora_mistral
from code_files.preprocess_data import Prepare_dataset_with_only_replace_mistral
# import argparse
from datasets import Dataset
from dotenv import load_dotenv
from utils import Custom_trainer

def main():    
    checkpoint = "mistralai/Mistral-7B-v0.1"
    load_dotenv()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model, tokenizer = Mistral_7B.create_model_and_tokenizer(checkpoint)
    model = Create_lora_mistral.create_lora(model, rank=4, dropout=0.05)
    # read and tokenized data
    path_trainset = "Datasets/vulgen_train_with_diff_lines_spaces.csv"
    path_testset = "Datasets/vulgen_test_with_diff_lines_spaces.csv"
    full_vulgen = True
    train, test = Prepare_dataset_with_only_replace_mistral.create_datasets(path_trainset, path_testset, full_vulgen=full_vulgen)
    
    def generate_prompt(sample, return_response=True):
        prompt = f"""<s>[INST] {sample['inputs']} [/INST] \n {sample['outputs']} </s>"""
        return prompt
    train['prompt'] = train.apply(generate_prompt, axis=1) 
    test['prompt'] = test.apply(generate_prompt, axis=1)
    max_length = 512 # This was an appropriate max length for my dataset

    def tokenize(dataset, tokenizer):
        result = tokenizer(
            dataset['prompt'].tolist(),
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
        result["labels"] = result["input_ids"].clone()
        return result
    
    tokenized_train_dataset = tokenize(train, tokenizer)
    tokenized_val_dataset = tokenize(test, tokenizer)
    tokenized_train_dataset = Dataset.from_dict(tokenized_train_dataset)
    tokenized_val_dataset = Dataset.from_dict(tokenized_val_dataset)
    # print(type(tokenized_train_dataset))
    # print(tokenized_train_dataset['input_ids'][0].shape)
    # config evaluation metrics
    metric = evaluate.load("sacrebleu")
    google_bleu = evaluate.load("google_bleu")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # print("preds: ", preds)
        # print("labels: ", labels)
        if isinstance(preds, tuple):
            preds = preds[0]
        # Convert preds to tensor if it's a NumPy array
        if isinstance(preds, np.ndarray):
            preds = torch.tensor(preds)
        preds = torch.argmax(torch.softmax(preds, dim=-1), dim=-1)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        print("decoded_preds[0]: ", decoded_preds[0])
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        print("decoded_labels: ", decoded_labels)
        print("decoded_preds: ", decoded_preds)
        decoded_preds[0] = decoded_preds[0].split("[/INST]")[1]
        decoded_labels[0] = decoded_labels[0].split("[/INST]")[1]
        #ScareBleu
        results = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"sacreBleu": results["score"]}
        #GoogleBlue
        results = google_bleu.compute(predictions=decoded_preds, references=decoded_labels)
        result["googleBleu"] = results["google_bleu"]
        #Accuracy
        count = 0
        for p, l in zip(decoded_preds, decoded_labels):
            if p == l[0]:
                count += 1
        total_tokens = len(decoded_labels)
        accuracy = count / total_tokens
        result['eval_accuracy'] = accuracy
        #Genaration length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        print("Computed metrics:", result)
        return result
    
    # def preprocess_logits_for_metrics(logits, labels):
    #     if isinstance(logits, tuple):
    #         logits = logits[0]
    #     return logits.argmax(dim=-1)

    # # config env varibles
    # NEPTUNE_API_TOKEN = os.environ.get("NEPTUNE_API_TOKEN")
    # NEPTUNE_PROJECT = os.environ.get("NEPTUNE_PROJECT")
    os.environ["NEPTUNE_API_TOKEN"] = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4Y2VlNTFhZC1hODJkLTQ4NzItOTE0MS0yZmNkNWY3ZWE0MTEifQ=='
    os.environ["NEPTUNE_PROJECT"] = 'zlotman/Localization-model'
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # create trainer object
    training_args = Seq2SeqTrainingArguments(
        output_dir="saved_models/Mistral",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        weight_decay=0.001,
        num_train_epochs=4,
        # predict_with_generate=True,
        bf16=True,
        tf32=True,
        # remove_unused_columns=False,
        logging_dir="TensorBoard",
        do_train=True,
        do_eval=True,
        logging_strategy='epoch',
        generation_max_length=512,
        # generation_num_beams=1,
        dataloader_num_workers=4,
        warmup_steps=57000,
        # report_to="no",
        lr_scheduler_type='linear',
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_runtime",
        greater_is_better=True
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()

if __name__ == "__main__":
    main()
    
    
# command = "NCCL_P2P_DISABLE='1' OMP_NUM_THREADS='1' accelerate launch --config_file accelerate_config_files/deepspeed_stage2.yaml code_files/fine_tuning/Mistral_7B/multi_gpus_simple_trainer.py"