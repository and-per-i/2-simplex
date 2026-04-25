import os
import sys
import torch
import pandas as pd
import time
from tqdm import tqdm
import re

# Aggiunge src e symbolic al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/symbolic")))

from src.models.student_model import StudentForCausalLM
from src.models.student_config import StudentConfig
from tokenizer.hf_tokenizer import load_tokenizer

# Import del motore Newclid locale
try:
    from newclid.api import GeometricSolverBuilder
    from newclid.jgex.formulation import JGEXFormulation
    from newclid.jgex.problem_builder import JGEXProblemBuilder
    from newclid.llm_input import new_problem_from_llm_aux_output
except ImportError as e:
    print(f"❌ Errore di importazione: {e}")
    sys.exit(1)

# CONFIGURAZIONE
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
if DEVICE == "mps":
    print("🚀 Metal (MPS) accelerazione attivata!")

TOKENIZER_PATH = "tokenizer/weights/geometry.757.model"
MODEL_DISTILLED = "checkpoints/checkpoint-96000"
MODEL_FINETUNED = "checkpoints/checkpoint-1600"
TEST_DATASET = "data/curriculum/stage_3_medium.parquet"
NUM_PROBLEMS = 50
PASS_K = 10

def load_model(path):
    if not os.path.exists(path):
        return None
    config = StudentConfig.from_pretrained(path)
    # Ottimizzazione M4: Usa float16 su MPS
    dtype = torch.float16 if DEVICE == "mps" else torch.float32
    model = StudentForCausalLM.from_pretrained(path, config=config, torch_dtype=dtype).to(DEVICE)
    model.eval()
    return model

def clean_ag_text(text):
    # 1. Rimuovi tag e metadati AG
    text = text.replace("<problem>", "").replace("</problem>", "")
    text = re.sub(r"\[\d+\]", "", text)
    
    if "?" in text:
        setup_part, goal_part = text.split("?", 1)
    else:
        setup_part, goal_part = text, ""

    mapping = {
        "coll": "on_line",
        "perp": "on_tline",
        "para": "on_pline",
        "midpoint": "midpoint",
        "circle": "on_circum",
        "on_circle": "on_circum",
        "cong": "eqdistance",
        "eqdistance": "eqdistance",
        "aconst": "aconst",
        "eqangle": "on_aline",
        "on_aline": "on_aline",
    }

    predicate_keywords = set(mapping.keys()) | {"cyclic", "on_line", "on_tline", "on_pline", "on_circum", "free"}

    new_clauses = []
    # Prima passata: identifica tutti i punti e le loro costruzioni
    for clause in setup_part.split(";"):
        clause = clause.strip()
        if not clause: continue
        
        if ":" in clause:
            points_part, const_part = clause.split(":", 1)
            target_points = [p.strip() for p in points_part.strip().split()]
            const_part = const_part.strip()
            
            if not const_part:
                for p in target_points:
                    new_clauses.append(f"{p} : free {p}")
                continue
            
            raw_words = const_part.split()
            # Estrai il predicato e i suoi argomenti
            pred = raw_words[0]
            args = raw_words[1:]
            
            mapped_pred = mapping.get(pred, pred)
            
            for target in target_points:
                # Alimenta Newclid con l'ordine corretto (target è sempre il primo)
                others = [a for a in args if a != target]
                
                if mapped_pred == "midpoint":
                    final_args = [target] + others[:2]
                elif mapped_pred == "on_line":
                    final_args = [target] + others[:2]
                elif mapped_pred in ["on_tline", "on_pline"]:
                    if len(others) >= 3:
                        final_args = [target, others[0], others[1], others[2]]
                    else:
                        final_args = [target] + others
                elif mapped_pred == "on_circum":
                    final_args = [target] + others[:3]
                elif mapped_pred == "eqdistance":
                    if len(others) >= 3:
                        final_args = [target, others[0], others[1], others[2]]
                    else:
                        final_args = [target] + others
                elif mapped_pred == "aconst":
                    if len(others) >= 4:
                        final_args = others[:3] + [target] + others[3:]
                    else:
                        final_args = [target] + others
                else:
                    final_args = [target] + others
                
                new_clauses.append(f"{target} : {mapped_pred} {' '.join(final_args)}")
        else:
            pts = clause.split()
            for p in pts:
                if re.match(r"^[a-z0-9_]+$", p):
                    new_clauses.append(f"{p} : free {p}")
                else:
                    new_clauses.append(clause)
            
    # Topological Reordering
    ordered_clauses = []
    defined_points = set()
    pending = []
    
    for clause in new_clauses:
        if " : " in clause:
            target, consts = clause.split(" : ", 1)
            target = target.strip()
            needed = set(re.findall(r"\b[a-z]\b", consts))
            if target in needed: needed.remove(target)
            
            if not needed or "free" in consts:
                ordered_clauses.append(clause)
                defined_points.add(target)
            else:
                pending.append({'clause': clause, 'target': target, 'needed': needed})
        else:
            ordered_clauses.append(clause)

    changed = True
    while changed:
        changed = False
        for i in range(len(pending)-1, -1, -1):
            item = pending[i]
            if item['needed'].issubset(defined_points):
                ordered_clauses.append(item['clause'])
                defined_points.add(item['target'])
                pending.pop(i)
                changed = True
    
    for item in pending:
        ordered_clauses.append(f"{item['target']} : free {item['target']}")
        defined_points.add(item['target'])

    setup_final = " ; ".join(ordered_clauses).strip()
    return f"{setup_final} ? {goal_part.strip()}" if goal_part else setup_final

def solve_with_newclid(problem_jgex, timeout=10):
    import signal
    from newclid.api import PythonDefault
    
    def handler(signum, frame):
        raise TimeoutError("Solver timeout")
        
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    
    try:
        builder = JGEXProblemBuilder().with_problem(problem_jgex)
        setup = builder.build(max_attempts_to_satisfy_goals_numerically=200)
        api_def = PythonDefault(use_sympy_ar=False)
        solver = GeometricSolverBuilder(api_default=api_def).build(setup)
        success = solver.run()
        signal.alarm(0)
        return success
    except TimeoutError:
        return False
    except Exception:
        signal.alarm(0)
        return False

def get_llm_suggestion(model, tokenizer, prompt, initial_problem, k=10):
    SYMBOLS = {
        "^": "eqangle3", "P": "on_pline", "T": "on_tline", "M": "midpoint", 
        "O": "on_circum", "I": "intersection_ll", "C": "eqdistance", "L": "on_line"
    }
    existing_points = set(initial_problem.points) if hasattr(initial_problem, 'points') else set()

    def get_safe_point(suggested):
        if suggested not in existing_points and len(suggested) == 1:
            return suggested
        for c in "pqrstuvwxyznm":
            if c not in existing_points: return c
        return "x_aux"

    def idx_to_letter(token):
        if token.isdigit():
            idx = int(token)
            if idx < 26: return chr(ord('a') + idx)
        return token

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=48, 
            do_sample=True, 
            temperature=0.9, 
            top_p=0.95, 
            num_return_sequences=k, 
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
    
    results = []
    seen_translated = set()
    for i in range(k):
        decoded = tokenizer.decode(outputs[i], skip_special_tokens=True)
        sugg = decoded.split("</pre_steps>")[-1].strip() if "</pre_steps>" in decoded else decoded[len(prompt):].strip()
        if not sugg: continue
        
        sugg = sugg.replace("▁", " ")
        tokens = re.split(r"[:=;(\n\s]+", re.sub(r"<[^>]+>|\[[^\]]+\]", "", sugg))
        tokens = [t.strip() for t in tokens if t.strip() and len(t.strip()) < 10]
        if not tokens: continue
        
        target_point = get_safe_point(idx_to_letter(tokens[0]))
        raw_const = tokens[1] if len(tokens) > 1 else "M"
        const_name = SYMBOLS.get(raw_const, "midpoint")
        
        potential_args = []
        for t in tokens[2:]:
            t_clean = idx_to_letter(t)
            if t_clean in existing_points and t_clean != target_point:
                potential_args.append(t_clean)
        
        # Newclid construction format alignment
        if const_name == "midpoint": args = [target_point] + potential_args[:2]
        elif const_name == "on_line": args = [target_point] + potential_args[:2]
        elif const_name in ["on_pline", "on_tline", "eqdistance"]: args = [target_point] + potential_args[:3]
        elif const_name == "on_circum": args = [target_point] + potential_args[:3]
        else: args = [target_point] + potential_args[:2]

        translated = f"{target_point} = {const_name} {' '.join(args)}"
        if translated not in seen_translated:
            results.append(translated)
            seen_translated.add(translated)
    return results

def run_benchmark():
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    print("🤖 Caricamento modelli (Precision: float16 su MPS)...")
    model_dist = load_model(MODEL_DISTILLED)
    model_ft = load_model(MODEL_FINETUNED)
    
    df = pd.read_parquet(TEST_DATASET).sample(NUM_PROBLEMS)
    stats = {"pure": 0, "distilled": 0, "finetuned": 0}

    print(f"\n🚀 Inizio Benchmark (N={NUM_PROBLEMS}, Pass@10)")
    print("-" * 60)

    for i, (idx, row) in enumerate(df.iterrows()):
        raw_text = row["question"]
        cleaned_text = clean_ag_text(raw_text)
        print(f"\n📝 PROB {i+1}/{NUM_PROBLEMS}:\n   [Cleaned]: {cleaned_text}")
        
        try:
            initial_problem = JGEXFormulation.from_text(cleaned_text)
        except Exception as e:
            print(f"   ⚠️ Errore parsing: {e}")
            continue

        # 1. Pure Newclid
        print("   🔹 Pure: ", end="", flush=True)
        if solve_with_newclid(initial_problem, timeout=10):
            print("✅")
            stats["pure"] += 1
            stats["distilled"] += 1
            stats["finetuned"] += 1
            continue
        print("❌", end="", flush=True)

        # 2. Distilled
        print(" | 🧪 Dist: ", end="", flush=True)
        found_dist = False
        if model_dist:
            prompt = f"<problem> {raw_text} </problem> <pre_steps> </pre_steps>"
            suggestions = get_llm_suggestion(model_dist, tokenizer, prompt, initial_problem, k=PASS_K)
            for sugg in suggestions:
                try:
                    aug_problem = new_problem_from_llm_aux_output(initial_problem, sugg, aux_tag="")
                    if solve_with_newclid(aug_problem, timeout=10):
                        found_dist = True
                        break
                except Exception: continue
        
        if found_dist:
            print("✅", end="", flush=True)
            stats["distilled"] += 1
        else:
            print("❌", end="", flush=True)

        # 3. Finetuned
        print(" | 🔥 Fine: ", end="", flush=True)
        found_ft = False
        if model_ft:
            prompt = f"<problem> {raw_text} </problem> <pre_steps> </pre_steps>"
            suggestions = get_llm_suggestion(model_ft, tokenizer, prompt, initial_problem, k=PASS_K)
            for sugg in suggestions:
                try:
                    aug_problem = new_problem_from_llm_aux_output(initial_problem, sugg, aux_tag="")
                    if solve_with_newclid(aug_problem, timeout=10):
                        found_ft = True
                        break
                except Exception: continue
        
        if found_ft:
            print("✅")
            stats["finetuned"] += 1
        else:
            print("❌")

    print("\n" + "="*50)
    print(f"📊 RISULTATI FINALI (N={NUM_PROBLEMS})")
    print("="*50)
    print(f"🧊 Pure Newclid:   {stats['pure']}/{NUM_PROBLEMS}")
    print(f"🧪 Distilled (96k): {stats['distilled']}/{NUM_PROBLEMS}")
    print(f"🔥 Finetuned (1.6k): {stats['finetuned']}/{NUM_PROBLEMS}")
    print("="*50)

if __name__ == "__main__":
    run_benchmark()
