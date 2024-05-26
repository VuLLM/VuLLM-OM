from transformers import TrainingArguments, DataCollatorForLanguageModeling
from trl import SFTTrainer
import torch
import numpy as np
import neptune
# from accelerate import Accelerator
import evaluate
import os
# import ..utils
import sys
sys.path.append('/sise/home/urizlo/VuLLM_One_Stage')
from utils import StarCoder2_7B, Create_lora_starCoder
from code_files.preprocess_data import Prepare_dataset_with_only_replace_only_encoder
# import argparse
from dotenv import load_dotenv
from datasets import Dataset



def main():    
    checkpoint = "bigcode/starcoder2-3b"
    load_dotenv()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    model, tokenizer = StarCoder2_7B.create_model_and_tokenizer(checkpoint)
    model.config.use_cache = True
    # print(model.config)

    # read and tokenized data
    path_trainset = "Datasets/vulgen_train_with_diff_lines_spaces.csv"
    path_testset = "Datasets/vulgen_test_with_diff_lines_spaces.csv"
    full_vulgen = True
    train, test = Prepare_dataset_with_only_replace_only_encoder.create_datasets(path_trainset, path_testset, full_vulgen=full_vulgen)
    train = Dataset.from_pandas(train)
    test= Dataset.from_pandas(test)
    # create lora adaptors
    model = Create_lora_starCoder.create_lora(model, rank=16, dropout=0.05)

    def generate_prompt(sample, return_response=True):
        prompt = f"""function: {sample['inputs']} \n instruction: \n {sample['outputs']}"""
        return [prompt]
    
    # config evaluation metrics
    metric = evaluate.load("sacrebleu")
    google_bleu = evaluate.load("google_bleu")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        
        if isinstance(preds, np.ndarray):
            preds = torch.tensor(preds)  
        else:
            preds = preds
        torch.cuda.empty_cache()
        preds = torch.argmax(torch.softmax(preds, dim=-1), dim=-1)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        # print("decoded_labels[0]: ", decoded_labels[0])
        # print("\n" + "\n")
        # print("decoded_preds[0]: ", decoded_preds[0])
        if "instruction" in decoded_preds[0][0]:
            print("yessssssssss")
            decoded_preds[0][0] = decoded_preds[0][0].split("instruction")[1]
            decoded_labels[0][0] = decoded_labels[0][0].split("instruction")[1]
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
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in decoded_preds[0][0]]
        result["gen_len"] = np.mean(prediction_lens)
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
    os.environ["NEPTUNE_API_TOKEN"] = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4Y2VlNTFhZC1hODJkLTQ4NzItOTE0MS0yZmNkNWY3ZWE0MTEifQ=='
    os.environ["NEPTUNE_PROJECT"] = 'zlotman/Localization-model'
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    

    # create trainer object
    training_args = TrainingArguments(
        output_dir="saved_models/StarCoder",
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
        bf16_full_eval=True,
        eval_accumulation_steps=1,
        tf32=True,
        label_names = ["labels"],
        # remove_unused_columns=False,
        # logging_dir="TensorBoard",
        do_train=True,
        do_eval=True,
        logging_strategy='epoch',
        # generation_max_length=810,
        # generation_num_beams=1,
        dataloader_num_workers=8,
        # warmup_steps=57000,
        report_to="neptune",
        lr_scheduler_type='linear',
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    max_seq_length = 2048
    trainer = SFTTrainer(
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
    # print(trainer.get_eval_dataloader().dataset)
    trainer.train()
    
if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Train a model with specific command line arguments.')
    # parser.add_argument('--path_trainset', type=str, default='Dataset_VulGen/vulgen_train_with_diff_lines_spaces.csv', help='Path to trainset csv file')
    # parser.add_argument('--path_testset', type=str, default='Dataset_VulGen/vulgen_train_with_diff_lines_spaces.csv', help='Path to testset csv file')
    # parser.add_argument('--full_vulgen', type=bool, default=False, help='Is trainset and test are from vulgen dataset?')
    # parser.add_argument('--output_dir', type=str, default='saved_models', help='Output directory for the saved model')
    # parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for training')
    # parser.add_argument('--batch_size_per_device', type=int, default=1, help='Batch size per device')
    # parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    # parser.add_argument('--generation_num_beams', type=int, default=1, help='Number of beams for generation')
    # args = parser.parse_args()
    main()
    
    
    # command = "NCCL_P2P_DISABLE='1' OMP_NUM_THREADS='1' accelerate launch --config_file accelerate_config_files/deepspeed_stage2.yaml code_files/fine_tuning/StarCoder2_7B/multi_gpus.py"