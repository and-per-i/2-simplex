import pandas as pd
from pathlib import Path
import random

def main():
    print("Costruzione dataset bilanciato per Fine-Tuning...")
    
    data_dir = Path("data/curriculum")
    stages = [
        "stage_1_very_easy.parquet",
        "stage_2_massive_easy.parquet",
        "stage_3_medium.parquet",
        "stage_4_mixed.parquet",
        "stage_5_difficult.parquet"
    ]
    
    # Preleviamo 5,000 campioni per ogni stage (o il massimo disponibile se sono meno)
    target_per_stage = 5000 
    
    dfs = []
    total_samples = 0
    
    for stage in stages:
        file_path = data_dir / stage
        if not file_path.exists():
            print(f"⚠️  Saltato {stage} (non trovato)")
            continue
            
        df = pd.read_parquet(file_path)
        avail = len(df)
        
        # Campiona senza rimpiazzo
        n_samples = min(target_per_stage, avail)
        df_sampled = df.sample(n=n_samples, random_state=42)
        
        dfs.append(df_sampled)
        total_samples += n_samples
        
        print(f"✅ {stage}: prelevati {n_samples:,} campioni (su {avail:,} totali)")

    # Unisci tutti i dataframe
    final_df = pd.concat(dfs, ignore_index=True)
    
    # Shuffle finale del dataset per non avere blocchi ordinati per difficoltà
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    output_path = Path("data/finetune_raw_balanced.parquet")
    output_path.parent.mkdir(exist_ok=True, parents=True)
    final_df.to_parquet(output_path, index=False)
    
    print("="*50)
    print(f"🎉 Dataset finale bilanciato creato: {output_path}")
    print(f"   Totale campioni: {len(final_df):,}")
    print("="*50)

if __name__ == "__main__":
    main()
