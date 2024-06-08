from transformers import TrainingArguments, DataCollatorForLanguageModeling
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import torch
import numpy as np
import neptune
# from accelerate import Accelerator
import evaluate
import os
# import ..utils
import sys
sys.path.append('/sise/home/urizlo/VuLLM_One_Stage')
from utils import DeepSeek_7B, Create_lora_starCoder, Custom_SFTTrainer
from code_files.preprocess_data import Prepare_dataset_with_only_replace_only_encoder
# import argparse
from dotenv import load_dotenv
from datasets import Dataset


def main():    
    checkpoint = "deepseek-ai/deepseek-coder-6.7b-base"
    load_dotenv()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model, tokenizer = DeepSeek_7B.create_model_and_tokenizer(checkpoint)

    # read and tokenized data
    path_trainset = "Datasets/vulgen_train_with_diff_lines_spaces.csv"
    path_testset = "Datasets/vulgen_test_with_diff_lines_spaces.csv"
    full_vulgen = True
    train, test = Prepare_dataset_with_only_replace_only_encoder.create_datasets(path_trainset, path_testset, full_vulgen=full_vulgen)
    train['prompt'] = train.apply(lambda row: f"""Function:\n{row['inputs']}\nInstruction:\n{row['outputs']}""", axis=1)
    test['prompt'] = test.apply(lambda row: f"""Function:\n{row['inputs']}\nInstruction:\n{row['outputs']}""", axis=1)
    train = Dataset.from_pandas(train)
    test= Dataset.from_pandas(test)
    max_seq_length = 1700
    # create lora adaptors
    model = Create_lora_starCoder.create_lora(model, rank=32, dropout=0.05)
    
    # Function to filter out long samples
    def filter_long_samples(example):
        inputs = tokenizer(example['prompt'], truncation=True, padding=False)
        return len(inputs['input_ids']) <= max_seq_length

# Apply the filter
    train = train.filter(filter_long_samples)
    test = test.filter(filter_long_samples)
    
    def generate_prompt(sample, return_response=True):
        prompt = sample['prompt']
        return prompt
    
    # config evaluation metrics
    metric = evaluate.load("sacrebleu")
    google_bleu = evaluate.load("google_bleu")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        gen_len_list = []

        if isinstance(preds, tuple):
            preds = preds[0]
            
        # Convert preds to tensor if it's a NumPy array
        if isinstance(preds, np.ndarray):
            preds = torch.tensor(preds)
        preds = torch.argmax(torch.softmax(preds, dim=-1), dim=-1)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Ensure labels are in numpy array format
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        # for i in range(len(decoded_preds)):
        #     if "Instruction" in decoded_preds[i] and "Instruction" in decoded_labels[i][0]:
        #         decoded_preds[i] = decoded_preds[i].split("Instruction")[1]
        #         decoded_labels[i] = decoded_labels[i][0].split("Instruction")[1]
        #         gen_len_list.append(len(tokenizer.encode(decoded_preds[i])))
        gen_len_list += [len(tokenizer.encode(pred)) for pred in decoded_preds]

        # print("decoded_labels[0]: ", decoded_labels[0])
        # print("\n" + "\n")
        # print("decoded_preds[0]: ", decoded_preds[0])

        # SacreBleu
        results = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"sacreBleu": results["score"]}

        # GoogleBleu
        results = google_bleu.compute(predictions=decoded_preds, references=decoded_labels)
        result["googleBleu"] = results["google_bleu"]

        # Accuracy
        count = 0
        for p, l in zip(decoded_preds, decoded_labels):
            if p == l:
                count += 1
        total_tokens = len(decoded_labels)
        accuracy = count / total_tokens
        result['eval_accuracy'] = accuracy

        # Generation length
        if gen_len_list:
            result["gen_len"] = round(np.mean(gen_len_list), 4)

        result = {k: round(v, 4) for k, v in result.items()}
        # print("Computed metrics:", result)
        return result
    
    # def preprocess_logits_for_metrics(logits, labels):
    #     if isinstance(logits, tuple):
    #         logits = logits[0]
    #     return logits.argmax(dim=-1)

    # # config env varibles
    # NEPTUNE_API_TOKEN = os.environ.get("NEPTUNE_API_TOKEN")
    # NEPTUNE_PROJECT = os.environ.get("NEPTUNE_PROJECT")
    # os.environ["NEPTUNE_API_TOKEN"] = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4Y2VlNTFhZC1hODJkLTQ4NzItOTE0MS0yZmNkNWY3ZWE0MTEifQ=='
    # os.environ["NEPTUNE_PROJECT"] = 'zlotman/Localization-model'
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    # Disable Neptune environment variables
    os.environ.pop("NEPTUNE_API_TOKEN", None)
    os.environ.pop("NEPTUNE_PROJECT", None)

    # create trainer object
    training_args = TrainingArguments(
        output_dir="saved_models/DeepSeek_7B",
        evaluation_strategy="epoch",
        learning_rate=8e-5,
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
        # generation_max_length=810,
        # generation_num_beams=1,
        dataloader_num_workers=4,
        # warmup_steps=57000,
        report_to="none",
        lr_scheduler_type='linear',
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_runtime",
        greater_is_better=True
    )

    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    response_template = "Instruction:"
    data_collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    trainer = Custom_SFTTrainer.Custom_SFTTrainer(
        model=model,
        args=training_args,
        dataset_batch_size=1,
        train_dataset=train,
        data_collator=data_collator,
        eval_dataset=test,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        formatting_func=generate_prompt,
        # preprocess_logits_for_metrics = preprocess_logits_for_metrics,
        compute_metrics=compute_metrics
    )

    trainer.train()

if __name__ == "__main__":
    main()
    
    
# command = "NCCL_P2P_DISABLE='1' OMP_NUM_THREADS='1' accelerate launch --config_file accelerate_config_files/deepspeed_stage2.yaml code_files/fine_tuning/DeepSeek_coder_6.7B/multi_gpus.py"