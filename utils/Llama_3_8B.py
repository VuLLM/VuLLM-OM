import torch
from torch import nn
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer
from accelerate import Accelerator
from peft import PeftModel, PeftConfig



def create_model_and_tokenizer(checkpoint):
    device_index = Accelerator().process_index
    device_map = {"": device_index}
    tokenizer = LlamaTokenizer.from_pretrained(checkpoint, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token    
    
    # tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    model = LlamaForCausalLM.from_pretrained(checkpoint,
                                            torch_dtype=torch.bfloat16,
                                            device_map=device_map)
    model.generate()
    # Add dropout to model layers
    # model.encoder.drop = nn.Dropout(p=0.05, inplace=False)
    # for i in range(20):
    #     model.encoder.h[i].attn.attn_dropout = nn.Dropout(p=0.05, inplace=False)
    #     model.encoder.h[i].attn.resid_dropout = nn.Dropout(p=0.05, inplace=False)
    # model.decoder.drop = nn.Dropout(p=0.05, inplace=False)
    # for i in range(31):
    #     model.decoder.transformer.h[i].attn.attn_dropout = nn.Dropout(p=0.05, inplace=False)
    #     model.decoder.transformer.h[i].attn.resid_dropout = nn.Dropout(p=0.05, inplace=False)
    #     model.decoder.transformer.h[i].mlp.dropout = nn.Dropout(p=0.05, inplace=False)
    return model, tokenizer

def create_model_and_tokenizer_one_GPU(checkpoint):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token    
    tokenizer.padding_side = 'right'
    # tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    model = LlamaForCausalLM.from_pretrained(checkpoint,
                                            torch_dtype=torch.bfloat16,
                                            device_map="auto")

    # Add dropout to model layers
    # model.encoder.drop = nn.Dropout(p=0.05, inplace=False)
    # for i in range(20):
    #     model.encoder.h[i].attn.attn_dropout = nn.Dropout(p=0.05, inplace=False)
    #     model.encoder.h[i].attn.resid_dropout = nn.Dropout(p=0.05, inplace=False)
    # model.decoder.drop = nn.Dropout(p=0.05, inplace=False)
    # for i in range(31):
    #     model.decoder.transformer.h[i].attn.attn_dropout = nn.Dropout(p=0.05, inplace=False)
    #     model.decoder.transformer.h[i].attn.resid_dropout = nn.Dropout(p=0.05, inplace=False)
    #     model.decoder.transformer.h[i].mlp.dropout = nn.Dropout(p=0.05, inplace=False)
    return model, tokenizer


def load_model_and_tokenizer(peft_model_id, from_hub=False, eval=False):
    device = "cuda"
    if from_hub:
        # device_index = Accelerator().process_index
        # device_map = {"": device_index}
        config = PeftConfig.from_pretrained(peft_model_id)
        model = LlamaForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True,
                                                        torch_dtype=torch.bfloat16,
                                                        trust_remote_code=True).to(device)
        tokenizer = LlamaTokenizer.from_pretrained(config.base_model_name_or_path)

        # Load the Lora model
        model = PeftModel.from_pretrained(model, peft_model_id)

    else:
        tokenizer = LlamaTokenizer.from_pretrained(peft_model_id)
        # device_index = Accelerator().process_index
        # device_map = {"": device_index}
        model = LlamaForCausalLM.from_pretrained(
                pretrained_model_name_or_path="saved_models/local_model/6B/second_round/checkpoint-7770",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True).to(device)    

    
    if not eval:
        # model.encoder.drop = nn.Dropout(p=0.05, inplace=False)
        # for i in range(20):
        #     model.encoder.h[i].attn.attn_dropout = nn.Dropout(p=0.05, inplace=False)
        #     model.encoder.h[i].attn.resid_dropout = nn.Dropout(p=0.05, inplace=False)
        # model.decoder.drop = nn.Dropout(p=0.05, inplace=False)
        # for i in range(31):
        #     model.decoder.transformer.h[i].attn.attn_dropout = nn.Dropout(p=0.05, inplace=False)
        #     model.decoder.transformer.h[i].attn.resid_dropout = nn.Dropout(p=0.05, inplace=False)

        for name, param in model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
        for param in model.parameters():
            if param.ndim == 1:
                # cast the small parameters (e.g. layernorm) to fp32 for stability
                param.data = param.data.to(torch.float32)
            # model.gradient_checkpointing_enable()  # reduce number of stored activations
            model.enable_input_require_grads()

        class CastOutputToFloat(nn.Sequential):
            def forward(self, x): return super().forward(x).to(torch.float32)
        model.decoder.lm_head = CastOutputToFloat(model.decoder.lm_head)
    return model, tokenizer