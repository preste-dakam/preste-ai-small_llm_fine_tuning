import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

# ─────────────────────────────────────────────
# 1. Configuration
# ─────────────────────────────────────────────
base_model_id = "./Qwen3-8B" 
test_file_path = "test_set.jsonl"
output_file_path = "base_model_inference_results.jsonl" # Saved here for comparison

# ─────────────────────────────────────────────
# 2. Load Base Model and Tokenizer 
# ─────────────────────────────────────────────
print(f"Loading base model '{base_model_id}' in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
model.eval()

print("Loading tokenizer from base model...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
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
# 4. Process FULL Dataset & Save Outputs
# ─────────────────────────────────────────────
# Count total lines so the progress bar is accurate
with open(test_file_path, "r", encoding="utf-8") as f:
    total_lines = sum(1 for _ in f)

print(f"\nStarting baseline inference on ALL {total_lines} examples...")

with open(test_file_path, "r", encoding="utf-8") as infile, \
     open(output_file_path, "w", encoding="utf-8") as outfile:
    
    for line in tqdm(infile, total=total_lines, desc="Extracting Data"):
        
        try:
            example = json.loads(line.strip())
            input_text = example["input_text"]
            expected_output = example.get("output_json", "{}") 
        except json.JSONDecodeError:
            continue

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

        try:
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
            
        except Exception as e:
            print(f"\n❌ ERROR during generation: {e}")
            # Instead of breaking, we continue to the next item so a single failure doesn't stop the whole file
            continue 

        # Robust Parsing
        try:
            start_idx = output_text.find('{')
            end_idx = output_text.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                clean_json_str = output_text[start_idx:end_idx+1]
                parsed_json = json.loads(clean_json_str)
                status = "success"
                thoughts = output_text[:start_idx].replace("<think>", "").replace("</think>", "").strip()
            else:
                parsed_json = output_text
                status = "failed_parse_no_brackets"
                thoughts = ""

        except json.JSONDecodeError:
            parsed_json = output_text 
            status = "failed_parse_bad_json"
            thoughts = ""

        result_record = {
            "input_text": input_text,
            "expected_json": expected_output,
            "predicted_json": parsed_json,
            "status": status,
            "thoughts": thoughts 
        }

        # Save and forcefully flush to disk
        outfile.write(json.dumps(result_record, ensure_ascii=False) + "\n")
        outfile.flush()  
        os.fsync(outfile.fileno()) 
        
print(f"\n✅ FULL baseline inference completed! Results saved to: {output_file_path}")