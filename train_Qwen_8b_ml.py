#pip install -U transformers trl peft datasets bitsandbytes accelerate mlflow
import torch
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

# --- Configuration ---
model_id = "Qwen/Qwen3-8B"
dataset_path = "medical_data.jsonl"
output_dir = "./qwen-medical-extractor"

# ─────────────────────────────────────────────
# MLflow Setup
# ─────────────────────────────────────────────
MLFLOW_EXPERIMENT = "qwen-medical-finetuning"
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

    # ── Training lifecycle ──────────────────────────────────────────

    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start_time = time.time()
        mlflow.log_params({
            # SFTConfig / TrainingArguments
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
            "max_grad_norm":                args.max_grad_norm,
            "warmup_ratio":                 args.warmup_ratio,
            "warmup_steps":                 args.warmup_steps,
            "weight_decay":                 args.weight_decay,
            "logging_steps":                args.logging_steps,
            "save_strategy":                str(args.save_strategy),
            "dataloader_num_workers":       args.dataloader_num_workers,
            "seed":                         args.seed,
            # SFT-specific
            "max_seq_length":               getattr(args, "max_length", None)
                                            or getattr(args, "max_seq_length", None),
            "dataset_text_field":           getattr(args, "dataset_text_field", None),
        })

    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.train_start_time
        mlflow.log_metrics({
            "total_training_time_seconds": total_time,
            "total_training_time_minutes": total_time / 60,
            "avg_step_time_seconds":       float(np.mean(self.step_times)) if self.step_times else 0.0,
        })

    # ── Epoch lifecycle ─────────────────────────────────────────────

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_time = time.time() - self.epoch_start_time
        epoch = int(state.epoch)
        mlflow.log_metrics({
            f"epoch_time_seconds": epoch_time,
        }, step=epoch)

    # ── Step-level logging ──────────────────────────────────────────

    def on_step_begin(self, args, state, control, **kwargs):
        self._step_start = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        step_time = time.time() - self._step_start
        self.step_times.append(step_time)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        step = state.global_step
        metrics = {}

        # Core training signals
        for key in ("loss", "grad_norm", "learning_rate", "epoch",
                    "train_loss", "train_runtime", "train_samples_per_second",
                    "train_steps_per_second"):
            if key in logs:
                metrics[key] = logs[key]

        # Throughput derived metrics
        if "train_samples_per_second" in logs:
            metrics["tokens_per_second_approx"] = (
                logs["train_samples_per_second"]
                * (getattr(args, "max_length", None) or getattr(args, "max_seq_length", 512))
            )

        # GPU memory (if CUDA available)
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved  = torch.cuda.memory_reserved(i)  / 1024**3
                metrics[f"gpu{i}_memory_allocated_gb"] = allocated
                metrics[f"gpu{i}_memory_reserved_gb"]  = reserved
                metrics[f"gpu{i}_memory_utilization"]  = allocated / reserved if reserved > 0 else 0.0

        if metrics:
            mlflow.log_metrics(metrics, step=step)

    # ── Checkpoint logging ──────────────────────────────────────────

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = os.path.join(
            args.output_dir,
            f"checkpoint-{state.global_step}"
        )
        mlflow.log_metric("checkpoint_saved_at_step", state.global_step, step=state.global_step)
        if os.path.isdir(checkpoint_dir):
            mlflow.log_artifacts(checkpoint_dir, artifact_path=f"checkpoints/step-{state.global_step}")


# ─────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
dataset = load_dataset("json", data_files=dataset_path, split="train")
#dataset = dataset.select(range(100))

def format_chat_template(example):
    system_prompt = (
        "Ви — медичний асистент. Ваше завдання — витягти структуровану "
        "інформацію з медичних інструкцій та повернути її виключно у форматі JSON."
    )
    messages = [
        {"role": "system",    "content": system_prompt},
        {"role": "user",      "content": example["input_text"]},
        {"role": "assistant", "content": example["output_json"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

formatted_dataset = dataset.map(format_chat_template)

# ─────────────────────────────────────────────
# Quantization (QLoRA)
# ─────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)

# ─────────────────────────────────────────────
# LoRA
# ─────────────────────────────────────────────
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# ─────────────────────────────────────────────
# Training Arguments
# ─────────────────────────────────────────────
training_args = SFTConfig(
    output_dir=output_dir,
    dataset_text_field="text",
    max_length=4096,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    fp16=False,
    bf16=True,
    report_to="none",   # we handle logging ourselves
)

# ─────────────────────────────────────────────
# Main MLflow Run
# ─────────────────────────────────────────────
with mlflow.start_run(run_name=f"sft-{model_id.split('/')[-1]}") as run:

    print(f"MLflow run ID: {run.info.run_id}")

    # ── Log model & experiment config ──────────────────────────────
    mlflow.log_params({
        "model_id":           model_id,
        "dataset_path":       dataset_path,
        "dataset_size":       len(formatted_dataset),
        "quantization":       "4bit-nf4",
        "double_quant":       True,
        "compute_dtype":      "bfloat16",
        # LoRA params
        "lora_r":             peft_config.r,
        "lora_alpha":         peft_config.lora_alpha,
        "lora_dropout":       peft_config.lora_dropout,
        "lora_bias":          peft_config.bias,
        "lora_target_modules": ",".join(peft_config.target_modules),
        "lora_task_type":     str(peft_config.task_type),
        # Hardware
        "cuda_available":     torch.cuda.is_available(),
        "gpu_count":          torch.cuda.device_count(),
        "gpu_name":           torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    })

    # ── Log dataset sample as artifact ─────────────────────────────
    sample_path = "/tmp/dataset_sample.json"
    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(formatted_dataset[:3]["text"], f, ensure_ascii=False, indent=2)
    mlflow.log_artifact(sample_path, artifact_path="dataset")

    # ── Log LoRA trainable parameter count ─────────────────────────
    peft_model = get_peft_model(model, peft_config)
    trainable, total = peft_model.get_nb_trainable_parameters()
    mlflow.log_params({
        "trainable_parameters":     trainable,
        "total_parameters":         total,
        "trainable_pct":            round(100 * trainable / total, 4),
    })
    # Revert to base model — SFTTrainer will apply peft_config itself
    model = peft_model.unload()

    # ── Trainer ────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted_dataset,
        peft_config=peft_config,
        args=training_args,
        processing_class=tokenizer,
        callbacks=[MLflowCallback()],
    )

    # ── Train ──────────────────────────────────────────────────────
    print("Starting fine-tuning...")
    train_result = trainer.train()

    # ── Log final summary metrics ───────────────────────────────────
    mlflow.log_metrics({
        "final_train_loss":         train_result.training_loss,
        "total_steps":              train_result.global_step,
        "total_flos":               train_result.metrics.get("total_flos", 0),
        "train_runtime_seconds":    train_result.metrics.get("train_runtime", 0),
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
        "train_steps_per_second":   train_result.metrics.get("train_steps_per_second", 0),
    })

    # ── Save adapters & log to MLflow ──────────────────────────────
    adapter_dir = f"{output_dir}/final_adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    mlflow.log_artifacts(adapter_dir, artifact_path="final_adapter")

    print(f"\nTraining complete.")
    print(f"Adapters saved to {adapter_dir}")
    print(f"MLflow run: {run.info.run_id}  |  experiment: {MLFLOW_EXPERIMENT}")