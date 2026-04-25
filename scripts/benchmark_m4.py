import sys
import torch
import torch.nn as nn
from pathlib import Path
import re
import time

# Setup paths per trovare il modello e il tokenizer
ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT_DIR / "modello_distillato"))

from student_progressive import StudentModelProgressive
from tokenizer.hf_tokenizer import load_tokenizer

def optimize_for_m4(model, device):
    """Ottimizzazioni specifiche per Apple Silicon M4 Metal"""
    model.to(device)
    # Su M4, la mezza precisione (FP16) è accelerata dai core AMX/GPU
    model.half() 
    model.eval()
    return model

@torch.inference_mode()
def generate_proof(model, tok, prompt, device, max_tokens=150):
    # Convertiamo il prompt in input_ids direttamente su MPS
    inputs = tok(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    eos_id = tok.eos_token_id
    
    start_time = time.time()
    
    for _ in range(max_tokens):
        # Inferenza ultra-veloce su Metal
        out = model(input_ids)
        logits = out["logits"] if isinstance(out, dict) else out
        
        # Prendiamo l'ultimo logit e passiamo al token successivo
        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        if next_token.item() == eos_id:
            break
            
    end_time = time.time()
    
    # Decodifica e pulizia output
    text = tok.decode(input_ids[0].tolist(), skip_special_tokens=True)
    
    # Mappa dei predicati per rendere l'output leggibile
    REVERSE_MAP = {
        '00': 'coll', '01': 'cong', '02': 'perp', '03': 'para',
        '04': 'midp', '05': 'eqangle', '06': 'eqratio', '07': 'sameclock',
        '08': 'sameside', '09': 'simtri', '10': 'contri', '11': 'cyclic', '12': 'circle'
    }
    for num, eng in REVERSE_MAP.items():
        text = re.sub(r'\b' + num + r'\b', eng, text)
        
    gen_time = end_time - start_time
    return text, gen_time

def main():
    print("="*80)
    print("🚀 M4 METAL BENCHMARK — Davide 8L Causal Model")
    print("="*80)

    # Forza l'uso di MPS (Metal)
    if not torch.backends.mps.is_available():
        print("❌ MPS non disponibile. Questo script richiede un Mac con Apple Silicon.")
        return
        
    device = torch.device("mps")
    
    # Cerca il modello nel path corretto (root o runs/)
    model_path_runs = ROOT_DIR / "runs" / "finetune_clean" / "pytorch_model_finetuned.bin"
    model_path_root = ROOT_DIR / "pytorch_model_finetuned.bin"
    
    if model_path_runs.exists():
        model_path = model_path_runs
    elif model_path_root.exists():
        model_path = model_path_root
    else:
        print(f"❌ Errore: Modello non trovato in {model_path_runs} o nella root.")
        return

    tok_path = ROOT_DIR / "modello_distillato" / "tokenizer" / "vocab.model"

    # 1. Caricamento Tokenizer e Modello
    print(f"📦 Caricamento modello da: {model_path.name}")
    print("⚙️  Inizializzazione acceleratori Metal (FP16)...")
    
    tok = load_tokenizer(str(tok_path), vocab_size=1024)
    model = StudentModelProgressive(vocab_size=1024, dim_hidden=384, num_layers=8, simplicial_layers=[3, 7])
    
    # Carica i pesi e ottimizza per M4
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = optimize_for_m4(model, device)

    categories = {
        "EASY": [
            ("Punti Medi", "a : ; b : ; c : ; d : 00 a b d 01 a d b d ; e : 00 a c e 01 a e c e ? 03 d e b c x00 "),
            ("Triangolo Isoscele", "a : ; b : ; c : 01 a b a c ; d : 00 b c d ? 05 a b c a c b x00 ")
        ],
        "MEDIUM": [
            ("Parallelogramma", "a : ; b : ; c : ; d : 03 a b c d 03 a d b c ? 01 a b c d x00 "),
            ("Altezza e Perpendicolari", "a : ; b : ; c : ; d : 00 b c d ; e : 02 a e b c ; f : 02 d f a e ? 03 d f b c x00 ")
        ],
        "HARD": [
            ("Talete / Rapporti", "a : ; b : ; c : ; d : 00 a b d ; e : 00 a c e 03 d e b c ? 06 a d a b a e a c x00 "),
            ("Circonferenza Inscritto", "a : ; b : ; c : 12 c1 a b c ; d : 00 a b d 11 a b c d ? 05 a c b a d b x00 ")
        ]
    }

    total_time = 0

    for cat, problems in categories.items():
        print(f"\n📂 [{cat}]")
        for name, prompt in problems:
            print(f"  • {name}...", end="", flush=True)
            proof, gen_time = generate_proof(model, tok, prompt, device)
            total_time += gen_time
            print(f" OK ({gen_time:.2f}s)")
            print(f"    RESULT: {proof}\n")

    print("="*80)
    print(f"✨ Benchmark completato in {total_time:.2f} secondi")
    print(f"🏎️  Il tuo M4 sta facendo volare la geometria neuro-simbolica!")
    print("="*80)

if __name__ == "__main__":
    main()
