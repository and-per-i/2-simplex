# Analisi Comparativa Sintassi: AlphaGeometry vs Newclid

## 1. Struttura del Problema

| Caratteristica | AlphaGeometry (Modello) | Newclid / JGEX (Motore) |
| :--- | :--- | :--- |
| **Dichiarazione Punti** | `a : ; b : ;` (Implicito) | `a : free a ; b : free b ;` (Esplicito) |
| **Costruzione** | `c : perp a b a c` | `c : on_tline c a b a` |
| **Ordine** | Spesso rilassato | Rigidamente topologico (niente forward-reference) |
| **Separatori** | `;` per clausole, `:` per definizioni | `;` per clausole, `:` tra punto e costruzione |

## 2. Mappatura dei Predicati (Dizionario di Traduzione)

| AlphaGeometry | Newclid (JGEX) | Note sui Parametri |
| :--- | :--- | :--- |
| `coll a b c` | `on_line c a b` | Il punto definito (`c`) è il primo argomento in Newclid. |
| `perp a b c d` | `on_tline d c a b` | Linea `ab` perpendicolare a `cd`. |
| `para a b c d` | `on_pline d c a b` | Linea `ab` parallela a `cd` passante per `c`. |
| `midpoint a b c` | `midpoint c a b` | `c` è il punto medio di `ab`. |
| `circle a b c d` | `on_circle d a b c` | `d` è sul cerchio passante per `a, b, c`. |
| `cong a b c d` | `eqdistance a b c d` | Segmento `ab` uguale a `cd`. |
| `eqangle a b c d e f g h` | `on_aline ...` | Rappresentazione complessa di angoli direzionati. |
| `aconst a b c d ratio` | `on_line_ratio ...` | Punto `d` su `ab` con rapporto specifico. |

## 3. Punti Critici di Allineamento

1. **Argomento Target**: In AG il punto può apparire ovunque nella clausola. In Newclid, la clausola deve iniziare con `target_point : construction_name target_point arg1 arg2...`.
2. **Coordinate Numeriche**: Newclid esegue un "numerical check" per validare il problema. Se i vincoli sono pochi o incoerenti, il check fallisce (`Numerical check failed`).
3. **Punti Liberi**: AG non specifica sempre quali punti sono liberi. Newclid richiede che ogni punto non costruito sia dichiarato come `free`.

## 4. Strategia di Allineamento per l'Inferenza

Per far sì che Newclid accetti i suggerimenti dell'LLM:
1. **Normalizzazione**: Ogni suggerimento dell'LLM (es. `p = midpoint a b`) deve essere trasformato in `p : midpoint p a b`.
2. **Iniezione Context**: Prima del suggerimento LLM, dobbiamo passare a Newclid l'intero setup del problema tradotto correttamente in formato JGEX.
3. **Validazione Topologica**: Se l'LLM suggerisce un punto che usa altri punti non ancora definiti, dobbiamo scartarlo o riordinarlo.
