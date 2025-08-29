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
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
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
    
    def preprocess_logits_for_metrics(logits, labels):
        labels_index = torch.argwhere(torch.bitwise_or(labels == 8241, labels == 3782))
        gold = torch.where(labels[labels_index[:, 0], labels_index[:, 1]] == 3782, 0, 1)
        labels_index[: , 1] = labels_index[: , 1] - 1
        logits = logits.softmax(dim=-1)
        logits = torch.softmax(logits[labels_index[:, 0], labels_index[:, 1]][:,[3782, 8241]], dim = -1)
        return logits[:, 1][2::3], gold[2::3]
    """
    
    # IDs and k
    YES_ID = 8241
    NO_ID  = 3782
    K = 5

    import torch
    import numpy as np

    def preprocess_logits_for_metrics(logits, labels):
        """
        Find all positions where labels are YES/NO using argwhere.
        Align to causal LM (use logits at t-1), extract P(YES) for each such position,
        and return flat arrays so Trainer can concat across steps.

        Returns (as "predictions" for compute_metrics):
        probs_yes_flat: float32 [N]   - P(YES_ID) at each matched position
        gold_yes_flat : int32   [N]   - 1 if label==YES_ID else 0
        batch_idx_flat: int32   [N]   - which batch item each element belongs to
        """
        with torch.no_grad():
            # logits: [B, T, V], labels: [B, T]
            B, T, V = logits.shape
            device = logits.device

            # indices of all YES/NO labels -> [N, 2] with columns [batch_idx, time_idx]
            idx = torch.argwhere((labels == YES_ID) | (labels == NO_ID))
            if idx.numel() == 0:
                # nothing to score
                empty = np.empty((0,), dtype=np.float32)
                empty_i = np.empty((0,), dtype=np.int32)
                return (empty, empty_i, empty_i)

            b = idx[:, 0]
            t = idx[:, 1]

            # gold: 1 if YES, 0 if NO
            gold = (labels[b, t] == YES_ID).to(torch.int32)

            # align to predicting logits at t-1; drop positions where t==0
            keep = t > 0
            b = b[keep]
            t = t[keep]
            gold = gold[keep]
            if b.numel() == 0:
                empty = np.empty((0,), dtype=np.float32)
                empty_i = np.empty((0,), dtype=np.int32)
                return (empty, empty_i, empty_i)

            # select raw logits at [b, t-1] -> [N, V]
            sel = logits[b, t - 1, :]  # raw (pre-softmax)

            # take the two target vocab columns and softmax once -> [N, 2]
            cols = torch.tensor([NO_ID, YES_ID], device=sel.device)
            two = sel.index_select(dim=1, index=cols)
            probs_yes = two.softmax(dim=1)[:, 1]  # P(YES_ID) -> [N]

            # flatten to numpy for Trainer concatenation
            return (
                probs_yes.detach().cpu().numpy().astype(np.float32),
                gold.detach().cpu().numpy().astype(np.int32),
                b.detach().cpu().numpy().astype(np.int32),   # batch index per element
            )


    def compute_metrics(eval_preds):
        """
        eval_preds[0] is the tuple returned above (probs_yes_flat, gold_yes_flat, batch_idx_flat).
        Compute mean recall@5 across sequences that contain at least one positive.
        """
        (probs_flat, gold_flat, batch_idx_flat), _ = eval_preds

        if len(probs_flat) == 0:
            return {"recall@5": 0.0}

        probs = np.asarray(probs_flat, dtype=np.float32)
        gold  = np.asarray(gold_flat,  dtype=np.int32)
        bidx  = np.asarray(batch_idx_flat, dtype=np.int32)

        recalls = []
        for b in np.unique(bidx):
            mask = (bidx == b)
            s = probs[mask]
            g = gold[mask]

            pos = np.flatnonzero(g == 1)
            if pos.size == 0:
                continue  # skip sequences with no positives

            k = min(K, s.size)
            # top-k indices by score (unordered but good for membership)
            topk_idx = np.argpartition(-s, k - 1)[:k]
            hits = np.intersect1d(topk_idx, pos).size
            recalls.append(hits / pos.size)

        return {"recall@5": float(np.mean(recalls)) if recalls else 0.0}

    
    os.environ["WANDB_DISABLED"] = "true"
    
    if sample > -1:
        if sample <= 128 :
            eval_step = 10
        else:
            eval_step = sample / 128 * 5
    
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
        callbacks = [EarlyStoppingCallback(early_stopping_patience=10)]
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
