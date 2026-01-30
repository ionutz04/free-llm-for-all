import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = os.environ.get("LLAMA_MODEL", "unsloth/Llama-3.2-3B-Instruct")

def load_model():
    print(f"Loading model: {MODEL_ID}", file=sys.stderr, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    return tokenizer, model

def chat_loop(tokenizer, model):
    print("Llama-3.2-3B CLI. Type 'exit' to quit.", flush=True)

    while True:
        try:
            user = input(">>> ").strip()
        except EOFError:
            break
        if user.lower() in {"exit", "quit"}:
            break
        if not user:
            continue

        # Simple prompt (no chat template, to avoid all dict nonsense)
        prompt = f"You are a concise technical assistant.\nUser: {user}\nAssistant:"
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
        ).to(model.device)  # inputs is a dict

        input_ids = inputs["input_ids"]      # (batch, seq_len) tensor
        attention_mask = inputs.get("attention_mask", None)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Slice off the prompt part
        generated_ids = output_ids[0, input_ids.shape[-1]:]
        generated = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        print(generated, flush=True)

if __name__ == "__main__":
    tok, mdl = load_model()
    chat_loop(tok, mdl)
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = os.environ.get("LLAMA_MODEL", "meta-llama/Llama-3.2-3B-Instruct")

def load_model():
    print(f"Loading model: {MODEL_ID}", file=sys.stderr, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    return tokenizer, model

def chat_loop(tokenizer, model):
    print("Llama-3.2-3B CLI. Type 'exit' to quit.", flush=True)
    history = []

    while True:
        try:
            user = input(">>> ").strip()
        except EOFError:
            break
        if user.lower() in {"exit", "quit"}:
            break
        if not user:
            continue

        messages = [
            {"role": "system", "content": "You are a concise technical assistant."},
        ]
        for u, a in history[-5:]:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": user})

        # 1) Build prompt tokens
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)  # input_ids is a tensor

        # 2) Generate using the tensor, not the dict
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        # 3) Decode only the new tokens
        generated = tokenizer.decode(
            output_ids[0][input_ids.shape[-1]:],
            skip_special_tokens=True,
        ).strip()

        print(generated, flush=True)
        history.append((user, generated))

if __name__ == "__main__":
    tok, mdl = load_model()
    chat_loop(tok, mdl)
