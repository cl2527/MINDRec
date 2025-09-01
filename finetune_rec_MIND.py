import os
os.environ['LD_LIBRARY_PATH'] = '/data/baokq/miniconda3/envs/alpaca_lora/lib/'
import sys
from typing import List

import numpy as np 
import fire
import torch
import transformers
from datasets import load_dataset
from transformers import EarlyStoppingCallback

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import (  # noqa: E402
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer  # noqa: F402
from sklearn.metrics import roc_auc_score

def train(
    # model/data params
    base_model: str = "",  # the only required argument
    train_data_path: str = "",
    val_data_path: str = "",
    output_dir: str = "./lora-alpaca",
    sample: int = -1,
    seed: int = 0,
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter

):
    print(
        f"Training Alpaca-LoRA model with params:\n"
        f"base_model: {base_model}\n"
        f"train_data_path: {train_data_path}\n"
        f"val_data_path: {val_data_path}\n"
        f"sample: {sample}\n"
        f"seed: {seed}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size
    # print(f"gradient_accumulation_steps: {gradient_accumulation_steps}")

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    ) 

    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        
        
        return result

 
    def generate_and_tokenize_prompt(data_point):
        tokenized = tokenize(generate_prompt(data_point))
        if not train_on_inputs:
            up = tokenize(generate_prompt({**data_point, "output": ""}), add_eos_token=False)
            n = len(up["input_ids"])
            tokenized["labels"] = [-100]*n + tokenized["labels"][n:]

        tokenized["user_id"] = data_point.get("user_id")

        return tokenized
    
    """
    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt
    """
    
    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)


    if train_data_path.endswith(".json"):  # todo: support jsonl
        train_data = load_dataset("json", data_files=train_data_path)
    else:
        train_data = load_dataset(train_data_path)
    
    if val_data_path.endswith(".json"):  # todo: support jsonl
        val_data = load_dataset("json", data_files=val_data_path)
    else:
        val_data = load_dataset(val_data_path)

    
    # train_data = train_data.shuffle(seed=42)[:sample] if sample > -1 else train_data
    # print(len(train_data))
    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    train_data["train"] = train_data["train"].shuffle(seed=seed).select(range(sample)) if sample > -1 else train_data["train"].shuffle(seed=seed)
    train_data["train"] = train_data["train"].shuffle(seed=seed)
    train_data = (train_data["train"].map(generate_and_tokenize_prompt))
    val_data = (val_data["train"].map(generate_and_tokenize_prompt))
    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True   
    
    """
    def compute_metrics(eval_preds):
        pre, labels = eval_preds
        auc = roc_auc_score(pre[1], pre[0])
        return {'auc': auc}
    """
    """
    def compute_metrics(eval_preds):
        # works with EvalPrediction or (preds, labels) tuple
        preds, lab = eval_preds

        # your preprocess returns (scores, gold)
        if isinstance(preds, (tuple, list)) and len(preds) == 2:
            scores = np.asarray(preds[0]).reshape(-1)   # prob(“Yes”)
            labels = np.asarray(preds[1]).reshape(-1)   # 0/1
        else:
            scores = np.asarray(preds).reshape(-1)
            labels = np.asarray(lab).reshape(-1)

        users = np.array(val_data["user_id"])  # from outer scope

        def recall_at_k(scores, labels, groups, k):
            recalls = []
            for u in np.unique(groups):
                idx = np.where(groups == u)[0]
                if idx.size == 0:
                    continue
                rel = labels[idx].astype(bool)
                pos = int(rel.sum())
                if pos == 0:
                    continue  # or append 0.0 if you prefer
                topk_idx = idx[np.argsort(-scores[idx])[: min(k, idx.size)]]
                hits = int(labels[topk_idx].sum())
                recalls.append(hits / pos)
            return float(np.mean(recalls)) if recalls else 0.0

        out = {f"recall@{k}": recall_at_k(scores, labels, users, k) for k in (1,3,5)}

        # optional: global AUC
        try:
            from sklearn.metrics import roc_auc_score
            out["auc"] = float(roc_auc_score(labels, scores))
        except Exception:
            pass
        return out
    """
        
    def compute_metrics(eval_preds):
      
        if hasattr(eval_preds, "predictions"):
            preds, lab = eval_preds.predictions, eval_preds.label_ids
        else:
            preds, lab = eval_preds

        if isinstance(preds, (tuple, list)) and len(preds) == 2:
            scores = np.asarray(preds[0]).reshape(-1)   # prob("Yes")
            labels = np.asarray(preds[1]).reshape(-1)   # 0/1
        else:
            scores = np.asarray(preds).reshape(-1)
            labels = np.asarray(lab).reshape(-1)

        groups = np.asarray(val_data["user_id"])

        uniq, inv = np.unique(groups, return_inverse=True)
        buckets = [np.where(inv == i)[0] for i in range(len(uniq))]

        def recall_at_k(k: int) -> float:
            vals = []
            for idx in buckets:
                n = idx.size
                if n == 0:
                    continue
                rel = labels[idx].astype(int)
                pos = int(rel.sum())
                if pos == 0:
                    continue
                # LOCAL order (0..n-1),
                local_order = np.argsort(-scores[idx], kind="mergesort")
                topk_local = local_order[: min(k, n)]
                hits = int(rel[topk_local].sum())
                vals.append(hits / pos)
            return float(np.mean(vals)) if vals else 0.0

        def ndcg_at_k(k: int) -> float:
            vals = []
            for idx in buckets:
                n = idx.size
                if n == 0:
                    continue
                rel = labels[idx].astype(int)
                if rel.sum() == 0:
                    continue
                local_order = np.argsort(-scores[idx], kind="mergesort")
                local_k = min(k, n)

                rel_sorted = rel[local_order][:local_k].astype(float)
                if rel_sorted.size == 0:
                    continue

                discounts = 1.0 / np.log2(np.arange(2, local_k + 2))
                dcg  = float(np.sum(rel_sorted * discounts))

                ideal = np.sort(rel)[::-1][:local_k].astype(float)
                idcg = float(np.sum(ideal * discounts))
                if idcg > 0:
                    vals.append(dcg / idcg)
            return float(np.mean(vals)) if vals else 0.0

        def mrr() -> float:
            vals = []
            for idx in buckets:
                n = idx.size
                if n == 0:
                    continue
                rel = labels[idx].astype(int)
                if rel.sum() == 0:
                    continue
                local_order = np.argsort(-scores[idx], kind="mergesort")
                ranked_rel  = rel[local_order]
                hits = np.flatnonzero(ranked_rel == 1)
                if hits.size > 0:
                    vals.append(1.0 / (hits[0] + 1))  # ranks are 1-based
            return float(np.mean(vals)) if vals else 0.0

        out = {}
        for k in (1, 3, 5):
            out[f"recall@{k}"] = recall_at_k(k)
            out[f"ndcg@{k}"]   = ndcg_at_k(k)
        out["mrr"] = mrr()

        try:
            from sklearn.metrics import roc_auc_score
            out["auc"] = float(roc_auc_score(labels, scores))
        except Exception:
            pass

        return out
    
    
    def preprocess_logits_for_metrics(logits, labels):
        """
        Original Trainer may have a memory leak. 
        This is a workaround to avoid storing too many tensors that are not needed.
        """
        labels_index = torch.argwhere(torch.bitwise_or(labels == 8241, labels == 3782))
        gold = torch.where(labels[labels_index[:, 0], labels_index[:, 1]] == 3782, 0, 1)
        labels_index[: , 1] = labels_index[: , 1] - 1
        logits = logits.softmax(dim=-1)
        logits = torch.softmax(logits[labels_index[:, 0], labels_index[:, 1]][:,[3782, 8241]], dim = -1)
        return logits[:, 1][2::3], gold[2::3]
    
    

    os.environ["WANDB_DISABLED"] = "true"

    
    eval_step = 4
    """
    if sample > -1:
        if sample <= 128 :
            eval_step = 1
        else:
            eval_step = sample / 128 * 5
    """
    
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=20,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=8,
            optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=eval_step,
            save_steps=eval_step,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=False,
            metric_for_best_model="eval_auc",
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to=None,
            # report_to="wandb" if use_wandb else None,
            # run_name=wandb_run_name if use_wandb else None,
            # eval_accumulation_steps=10,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        #callbacks = [EarlyStoppingCallback(early_stopping_patience=10)]
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    """
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))
    """
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    print('evaluation before training')
    metrics0 = trainer.evaluate(metric_key_prefix="eval")
    metrics0["epoch"] = 0.0
    trainer.log(metrics0)          # logs to state.log_history and any reporters
    trainer.save_metrics("eval_init", metrics0)  # writes to output_dir

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  # noqa: E501

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  # noqa: E501

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""


if __name__ == "__main__":
    fire.Fire(train)
