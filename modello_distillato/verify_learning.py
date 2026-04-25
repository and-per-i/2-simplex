"""
Verifica Finale post-Finetuning
===============================

Esegue una generazione di prova sul modello appena fine-tunato
per verificare che abbia appreso il nuovo formato "clean" dei token.
"""

import sys
import torch
from pathlib import Path

# Setup paths per importare dal progetto
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from student_progressive import StudentModelProgressive
from tokenizer.hf_tokenizer import load_tokenizer

def generate_text(model, input_ids, max_new_tokens, eos_token_id):
    """Semplice generazione auto-regressiva (greedy decoding)"""
    device = input_ids.device
    
    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(input_ids)
            
        logits = out["logits"] if isinstance(out, dict) else out
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        if next_token.item() == eos_token_id:
            break
            
    return input_ids

def main():
    print("="*60)
    print("  VERIFICA LEARNING — Modello 8 Layer (Clean Syntax)")
    print("="*60)

    # 1. Trova l'ultimo modello fine-tunato
    model_path = SCRIPT_DIR.parent / "runs" / "finetune_clean" / "pytorch_model_finetuned.bin"
    if not model_path.exists():
        print(f"❌ Modello non trovato in: {model_path}")
        print("Assicurati di aver scaricato la cartella runs/ dal cloud o modifica il path.")
        return

    tokenizer_path = SCRIPT_DIR / "tokenizer" / "vocab.model"
    
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Caricamento
    print(f"\n📦 Caricamento Tokenizer...")
    tok = load_tokenizer(str(tokenizer_path), vocab_size=1024)

    print(f"📦 Caricamento Modello a 8 Layer...")
    model = StudentModelProgressive(
        vocab_size=1024,
        dim_hidden=384,
        num_layers=8,
        simplicial_layers=[3, 7]
    )
    
    sd = torch.load(model_path, map_location="cpu")
    model.load_state_dict(sd)
    model.to(device)
    model.eval()

    # 3. Test Prompts (usando i token nativi 00, 01, etc. e spazio finale!)
    prompts = [
        # Test 1: Triangolo con punti medi -> segmento parallelo alla base
        "a : ; b : ; c : ; d : 00 a b d 01 a d b d ; e : 00 a c e 01 a e c e ? 03 d e b c ",
        
        # Test 2: Angoli alla base di triangolo isoscele
        "a : ; b : ; c : 01 a b a c ; d : 00 b c d ? 05 a b c a c b "
    ]
    
    # Mappa inversa per rendere l'output leggibile
    REVERSE_MAP = {
        '00': 'coll', '01': 'cong', '02': 'perp', '03': 'para',
        '04': 'midpoint', '05': 'eqangle', '06': 'eqratio', '07': 'sameclock',
        '08': 'sameside', '09': 'simtri', '10': 'contri', '11': 'cyclic', '12': 'circle'
    }

    print("\n" + "—"*60)
    for i, p in enumerate(prompts):
        print(f"\n🎯 TEST {i+1}:")
        print(f"PROMPT: {p}")
        
        inputs = tok(p, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        
        print("\n⏳ Generazione in corso...")
        out_ids = generate_text(
            model=model, 
            input_ids=input_ids, 
            max_new_tokens=150, 
            eos_token_id=tok.eos_token_id
        )
            
        generated_text = tok.decode(out_ids[0].tolist(), skip_special_tokens=True)
        
        # Traduci di nuovo nei termini inglesi
        import re
        for num, eng in REVERSE_MAP.items():
            generated_text = re.sub(r'\b' + num + r'\b', eng, generated_text)
            
        print("\n✨ OUTPUT GENERATO:")
        print(generated_text)
        print("—"*60)


if __name__ == "__main__":
    main()
