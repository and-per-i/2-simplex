import torch
import sys
import os
from src.models.student_model import StudentForCausalLM
from src.models.student_config import StudentConfig
from tokenizer.hf_tokenizer import load_tokenizer

def test_native_inference():
    # Configurazione percorsi
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model_path = "checkpoints/checkpoint-1600"
    tokenizer_path = "tokenizer/weights/geometry.757.model"
    
    if not os.path.exists(model_path):
        print(f"Modello non trovato in {model_path}")
        return

    tokenizer = load_tokenizer(tokenizer_path)
    config = StudentConfig.from_pretrained(model_path)
    
    print(f"--- Caricamento Modello su {device} ---")
    model = StudentForCausalLM.from_pretrained(
        model_path, 
        config=config,
        torch_dtype=torch.float16 if device == "mps" else torch.float32
    ).to(device)
    model.eval()

    # PROMPT IN PURO NEWCLID (JGEX)
    # Nessun tag <problem>, nessuna numerazione [000]
    prompt = "a : free a ; b : free b ; c : free c ; d : midpoint d a b ; e : midpoint e a c ? para d e b c"
    
    print(f"\nPROMPT (Native Newclid): {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    print("\nGenerazione in corso...")
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs, 
            max_new_tokens=64,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    print("\n================ RAW OUTPUT ================")
    print(decoded)
    print("============================================\n")

if __name__ == "__main__":
    test_native_inference()
