import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm # Great for showing a progress bar!

# ─────────────────────────────────────────────
# 1. Configuration
# ─────────────────────────────────────────────
base_model_id = "./Qwen3-8B" 
adapter_path = "./qwen-medical-extractor/trial_1/final_adapter" # Update to your best trial
test_file_path = "test_set.jsonl"
output_file_path = "inference_results.jsonl" # Where the outputs will be saved

# ─────────────────────────────────────────────
# 2. Load Model and Tokenizer
# ─────────────────────────────────────────────
print("Loading base model in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

print("Attaching LoRA adapter...")
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(adapter_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ─────────────────────────────────────────────
# 3. System Prompt
# ─────────────────────────────────────────────
system_prompt = (
    "Ти — медичний асистент. Твоє завдання — витягти структуровану "
    "інформацію з медичних інструкцій та повернути її виключно у форматі JSON.\n\n"
    "Використовуй наступну схему для формування JSON:\n"
    "- \"Торгівельне найменування\" (string): Повна торгова назва лікарського засобу.\n"
    "- \"Форма випуску\" (string): Лікарська форма, дозування та деталі пакування.\n"
    "- \"Умови відпуску\" (string): Умови відпуску з аптеки.\n"
    "- \"Склад (діючі)\" (string): Інформація про діючу речовину (речовини) та її кількість.\n"
    "- \"Фармакотерапевтична група\" (string): Повна назва фармакотерапевтичної групи.\n"
    "- \"Термін придатності\" (string): Термін зберігання лікарського засобу.\n"
    "- \"Виробник 1: назва українською\" (string): Офіційна назва компанії-виробника."
)

# ─────────────────────────────────────────────
# 4. Process Dataset & Save Outputs
# ─────────────────────────────────────────────
# Count lines for the progress bar
with open(test_file_path, "r", encoding="utf-8") as f:
    total_lines = sum(1 for _ in f)

print(f"\nStarting inference on {total_lines} examples...")

with open(test_file_path, "r", encoding="utf-8") as infile, \
     open(output_file_path, "w", encoding="utf-8") as outfile:
    
    for line in tqdm(infile, total=total_lines, desc="Extracting Data"):
        example = json.loads(line)
        input_text = example["input_text"]
        expected_output = example.get("output_json", "{}") # If you want to compare later

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text}
        ]

        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Attempt to parse the output as JSON to ensure it's valid
        try:
            parsed_json = json.loads(output_text)
            status = "success"
        except json.JSONDecodeError:
            # If the model hallucinates outside of JSON format, save the raw text anyway
            parsed_json = output_text 
            status = "failed_parse"

        # Create a record containing the input, the ground truth, and the model's prediction
        result_record = {
            "input_text": input_text,
            "expected_json": expected_output,
            "predicted_json": parsed_json,
            "status": status
        }

        # Write the record to the new file immediately
        outfile.write(json.dumps(result_record, ensure_ascii=False) + "\n")
        
print(f"\n✅ All inferences completed and saved to: {output_file_path}")