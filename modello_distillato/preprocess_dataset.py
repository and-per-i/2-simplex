"""
Preprocessing Dataset per Fine-Tuning Corretto
===============================================

Converte il formato XML del dataset Parquet in formato pulito
compatibile con il tokenizer geometry.757.model.

PRIMA (formato attuale):
  <problem> a : ; b : ; cong a b c d [000] ? simtrir a b d </problem>
  <proof> eqratio a b ... [003] a00 [000] ; </proof>

DOPO (formato pulito):
  a ; b ; cong a b c d ; ? simtrir a b d
  eqratio a b ... a00 ;

Usage (sul cloud):
    python preprocess_dataset.py \
        --input data/parquets/train0901-00000-of-00003.parquet \
        --output data/parquets/train_clean.parquet \
        --preview 5
"""

import re
import argparse
from pathlib import Path


def clean_sample(question: str, solution: str) -> str:
    """
    Rimuove tag XML e brackets, producendo testo pulito
    compatibile con il tokenizer geometry.757.model.
    """
    # 1. Rimuovi tag XML strutturali (qualsiasi tag <qualcosa> o </qualcosa>)
    text = question + " " + solution
    text = re.sub(r'</?[a-zA-Z_]+>', '', text)

    # 2. Rimuovi index brackets [000], [001], ..., [999]
    #    Questi erano riferimenti a step precedenti.
    #    In formato lineare non servono (l'ordine sequenziale li sostituisce).
    text = re.sub(r'\[\d{3}\]', '', text)

    # 3. Rimuovi "sameclock", "sameside" e altri predicati di check numerico
    #    (sono verifiche interne, non parte della prova formale)
    text = re.sub(r'(sameclock|sameside|samecirc|diff)\s+[a-z\s]+;', '', text)

    # 4. Normalizza whitespace multipli → singolo spazio
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n+', '\n', text)

    # 5. Rimuovi righe vuote
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    return ' '.join(lines)


def validate_clean_sample(text: str) -> bool:
    """Verifica che il campione pulito sia valido."""
    # Deve contenere almeno una dichiarazione di punto (es. "a ;" oppure "a : ;")
    if not re.search(r'\b[a-z]\s*:', text) and not re.search(r'\b[a-z]\s*;', text):
        return False
    # Deve contenere almeno un ';' (separatore step)
    if ';' not in text:
        return False
    # Non deve contenere XML residuo
    if '<' in text or '>' in text:
        return False
    return True


def preview_samples(df, n=3):
    """Mostra un confronto prima/dopo per N campioni."""
    print("\n" + "="*70)
    print("PREVIEW TRASFORMAZIONE")
    print("="*70)
    for i in range(min(n, len(df))):
        row = df.iloc[i]
        q = str(row.get('question', ''))
        s = str(row.get('solution', ''))
        clean = clean_sample(q, s)

        print(f"\n--- Campione {i+1} ---")
        print(f"BEFORE: {(q + ' ' + s)[:200]}...")
        print(f"AFTER:  {clean[:200]}...")
        print(f"Valid:  {'✅' if validate_clean_sample(clean) else '❌'}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,
                        default='data/parquets/train0901-00000-of-00003.parquet',
                        help='Path al Parquet originale')
    parser.add_argument('--output', type=str,
                        default='data/parquets/train_clean.parquet',
                        help='Path output Parquet pulito')
    parser.add_argument('--preview', type=int, default=3,
                        help='Numero campioni da mostrare in preview')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Limita il numero di campioni (debug)')
    args = parser.parse_args()

    print(f"📦 Caricamento: {args.input}")
    import pandas as pd
    df = pd.read_parquet(args.input)
    print(f"✅ Caricati {len(df)} campioni")
    print(f"   Colonne: {df.columns.tolist()}")

    if args.max_samples:
        df = df.head(args.max_samples)
        print(f"   (Limitato a {args.max_samples} per debug)")

    # Preview prima di processare
    if args.preview > 0:
        preview_samples(df, args.preview)

    # Processing
    print(f"\n🔄 Preprocessing {len(df)} campioni...")
    cleaned_texts = []
    skipped = 0

    for i, row in df.iterrows():
        q = str(row.get('question', ''))
        s = str(row.get('solution', ''))
        clean = clean_sample(q, s)

        if validate_clean_sample(clean):
            cleaned_texts.append(clean)
        else:
            cleaned_texts.append(None)
            skipped += 1

        if (i + 1) % 100_000 == 0:
            print(f"  Processati: {i+1}/{len(df)} ({skipped} skippati)")

    # Crea DataFrame output
    df_out = pd.DataFrame({
        'text': cleaned_texts,
        'original_idx': range(len(df))
    })
    df_out = df_out[df_out['text'].notna()]

    print(f"\n📊 Risultato:")
    print(f"  - Totale: {len(df)}")
    print(f"  - Validi: {len(df_out)} ({len(df_out)/len(df)*100:.1f}%)")
    print(f"  - Skippati: {skipped}")

    # Lunghezza media tokens (stima)
    avg_len = df_out['text'].str.split().str.len().mean()
    print(f"  - Lunghezza media parole: {avg_len:.0f}")

    # Salva
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(args.output, index=False)
    print(f"\n✅ Salvato: {args.output} ({output_path.stat().st_size / 1e6:.1f} MB)")

    # Campioni di output per verifica finale
    print("\n📋 Ultimi 2 campioni processati:")
    for text in df_out['text'].tail(2):
        print(f"  → {text[:200]}")


if __name__ == '__main__':
    main()
