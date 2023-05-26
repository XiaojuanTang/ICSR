#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from dataclasses import dataclass, field
from typing import Dict, Optional
import transformers
import torch
from tqdm import tqdm
import pickle

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{} is the {}\n\n### Input:\n{}\n\n### Response:"
    )


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="insert_sem")
    data_path: Optional[str] = field(default="part0.bin")
    use_sym: Optional[bool] = field(default=False)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def train():
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        device_map={'':'cuda:0'}
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        device="cuda:0"
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    sym=model_args.use_sym
    with open(model_args.data_path, 'rb') as f:
        data_list = pickle.load(f)
    insert_ranking_list=[]
    edit_ranking_list=[]
    insert_success_list=[]
    for data in data_list:
        for key,value in tqdm(data.items()):
            insert_list = list(value['insert'])
            edit_list = list(value['edit'])
            back_list = list(value['back'])
            prefixes=PROMPT.format(key[0],key[2 if sym else 1], ', '.join([n for n in insert_list+edit_list+back_list]))
            #print(prefixes)
            prefix_lens = len(tokenizer(prefixes)["input_ids"])
            Response_tok = [tokenizer(f" {n}")["input_ids"][2:] for n in insert_list+edit_list+back_list]
            Response_len = [len(n) for n in Response_tok]
            prompt_tok = tokenizer([f"{PROMPT.format(key[0],key[2 if sym else 1], ', '.join([n for n in insert_list+edit_list+back_list]))} {suffix}"for suffix in insert_list+edit_list+back_list],padding=True,return_tensors="pt")
            with torch.no_grad():
                output = model.model(**prompt_tok)
                logits = model.lm_head(output[0])
            probs = torch.zeros(logits.size(0))
            for i in range(logits.size(0)):
                cur_toks = Response_tok[i]
                cur_len = Response_len[i]
                # Compute suffix probabilities
                for j in range(cur_len):
                    cur_tok = cur_toks[j]
                    probs[i] += -torch.nn.functional.log_softmax(logits[i, prefix_lens + j - 1, :], dim=0)[cur_tok].item()
                probs[i] /= cur_len
            #for res in probs.argsort()[:len(insert_list)]:
            #    print([n for n in insert_list+edit_list+back_list][res])

            current_insert_ranking=[]
            current_edit_ranking=[]
            current_insert_success=[]
            for i in range(len(insert_list)):
                current_insert_ranking.append(int(torch.gt(probs[i], probs[len(insert_list):]).sum()))
                current_edit_ranking.append(int(torch.gt(probs[i+len(insert_list)] ,torch.cat([probs[:len(insert_list)], probs[len(insert_list)+len(edit_list):]])).sum()))
                current_insert_success.append(int(torch.gt(probs[i] ,probs[len(insert_list):len(insert_list)+len(edit_list)]).sum()))
            insert_ranking_list.append(min(current_insert_ranking))
            edit_ranking_list.append(min(current_edit_ranking))
            insert_success_list.append(min(current_insert_success))
            '''print("Insert: ", insert_list)
            print("Edit: ", edit_list)
            print('Current Insert ranking: ',min(current_insert_ranking))
            print('Current Edit ranking: ', min(current_edit_ranking))
            print('Current Insert success ranking: ', min(current_insert_success))'''
    print('Total Insert MRR: ',torch.mean(1/(torch.Tensor(insert_ranking_list)+1)))
    print('Total Edit MRR: ', torch.mean(1/(torch.Tensor(edit_ranking_list)+1)))
    print('Total Insert Success: ', torch.mean(1/(torch.Tensor(insert_success_list)+1)))
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

if __name__ == "__main__":
    train()
