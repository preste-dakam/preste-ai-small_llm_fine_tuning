
import torch
import optuna 
import gc
import mlflow
import mlflow.pytorch
import numpy as np
import os
import json
import time
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

model_id = "./Qwen3-8B" # "Qwen/Qwen3-8B" if you have access to the internet
dataset_path = "train_set.jsonl"
output_dir = "./qwen-medical-extractor"

# ─────────────────────────────────────────────
# MLflow Setup & Callback
# ─────────────────────────────────────────────
MLFLOW_EXPERIMENT = "qwen-medical-finetuning-optuna"
mlflow.set_experiment(MLFLOW_EXPERIMENT)

class MLflowCallback(TrainerCallback):
    """
    Custom callback that hooks into every Trainer event and logs
    metrics, system stats, and artefacts to MLflow.
    """
    def __init__(self):
        self.train_start_time = None
        self.epoch_start_time = None
        self.step_times = []

    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start_time = time.time()
        mlflow.log_params({
            "output_dir":                   args.output_dir,
            "per_device_train_batch_size":  args.per_device_train_batch_size,
            "gradient_accumulation_steps":  args.gradient_accumulation_steps,
            "effective_batch_size":         args.per_device_train_batch_size * args.gradient_accumulation_steps,
            "learning_rate":                args.learning_rate,
            "lr_scheduler_type":            str(args.lr_scheduler_type),
            "num_train_epochs":             args.num_train_epochs,
            "optim":                        args.optim,
            "fp16":                         args.fp16,
            "bf16":                         args.bf16,
            "logging_steps":                args.logging_steps,
            "seed":                         args.seed,
            "max_seq_length":               getattr(args, "max_length", None) or getattr(args, "max_seq_length", None),
        })

    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.train_start_time
        mlflow.log_metrics({
            "total_training_time_seconds": total_time,
            "avg_step_time_seconds":       float(np.mean(self.step_times)) if self.step_times else 0.0,
        })

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_time = time.time() - self.epoch_start_time
        epoch = int(state.epoch)
        mlflow.log_metrics({f"epoch_time_seconds": epoch_time}, step=epoch)

    def on_step_begin(self, args, state, control, **kwargs):
        self._step_start = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        step_time = time.time() - self._step_start
        self.step_times.append(step_time)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs: return
        step = state.global_step
        metrics = {}
        for key in ("loss", "eval_loss", "grad_norm", "learning_rate", "epoch"):
            if key in logs:
                metrics[key] = logs[key]

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved  = torch.cuda.memory_reserved(i)  / 1024**3
                metrics[f"gpu{i}_memory_allocated_gb"] = allocated
                metrics[f"gpu{i}_memory_reserved_gb"]  = reserved

        if metrics:
            mlflow.log_metrics(metrics, step=step)

# --- Load Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ─────────────────────────────────────────────
# Dataset Processing & Splitting
# ─────────────────────────────────────────────
dataset = load_dataset("json", data_files=dataset_path, split="train")
# Add this line right below if you want to test your script on a small amount of data:
#dataset = dataset.select(range(100))

# deterministic 90/10 split 
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

def format_chat_template(example):
    system_prompt = (
        "Ти — медичний асистент. Твоє завдання — витягти структуровану "
        "інформацію з медичних інструкцій та повернути її виключно у форматі JSON."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example["input_text"]},
        {"role": "assistant", "content": example["output_json"]}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

train_dataset = train_dataset.map(format_chat_template)
eval_dataset = eval_dataset.map(format_chat_template)

# ─────────────────────────────────────────────
# Optuna Objective Function
# ─────────────────────────────────────────────
def objective(trial):
    print(f"\n--- Starting Trial {trial.number} ---")
    
    # Start a nested MLflow run for each individual Optuna trial
    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True) as run:
        
        lr = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
        batch_size = trial.suggest_categorical("per_device_train_batch_size", [1, 2])
        epochs = trial.suggest_int("num_train_epochs", 2, 4)
        # Use line right below if you want to test quickly your script (comment the line up)
        #epochs = 1

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )

        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        # Log overarching parameters specific to this run
        mlflow.log_params({
            "model_id": model_id,
            "trial_number": trial.number,
            "train_size": len(train_dataset),
            "eval_size": len(eval_dataset)
        })

        trial_output_dir = f"{output_dir}/trial_{trial.number}"
        training_args = SFTConfig(
            output_dir=trial_output_dir,
            dataset_text_field="text",
            max_length=4096,
            eval_strategy="epoch", 
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            gradient_accumulation_steps=4,
            fp16=False,
            bf16=True,
            logging_steps=10,
            report_to="none" # Disable default MLflow to prevent duplicating logs with our custom callback **
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
            args=training_args,
            processing_class=tokenizer,
            callbacks=[MLflowCallback()], #Injecting the MLflow Callback 
        )

        # Train and Evaluate
        trainer.train()
        eval_results = trainer.evaluate()
        
        # Save the trial's adapter and tokenizers, then log them to MLflow
        adapter_dir = f"{trial_output_dir}/final_adapter"
        trainer.model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        mlflow.log_artifacts(adapter_dir, artifact_path="final_adapter")
        
        # Log the ultimate evaluation loss to MLflow before destroying the model
        mlflow.log_metric("final_eval_loss", eval_results["eval_loss"])
        
        # Memory Cleanup
        del trainer
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
        return eval_results["eval_loss"]

# ─────────────────────────────────────────────
# Optuna Search Execution
# ─────────────────────────────────────────────
print("Initializing Optuna Hyperparameter Search...")

#Group all Optuna trials into a single parent MLflow run
with mlflow.start_run(run_name="optuna_hyperparameter_search"):
    study = optuna.create_study(direction="minimize") 
    study.optimize(objective, n_trials=5) 
    # Use line right below if you want to test quickly your script (comment the line up)
    #study.optimize(objective, n_trials=1)

    # Output the ultimate results
    print("\n=== HYPERPARAMETER SEARCH COMPLETE ===")
    print("Best Hyperparameters found:")
    for key, value in study.best_params.items():
        print(f" - {key}: {value}")
    print(f"Best validation loss achieved: {study.best_value}")
    
    # Log the best metrics to the parent MLflow run
    mlflow.log_params({"best_" + k: v for k, v in study.best_params.items()})
    mlflow.log_metric("best_validation_loss", study.best_value)