---
name: orgonite-kb
description: >
  Query the Orgonite Master knowledge base (also referred to as the Olson KB) —
  a local corpus of esoteric and technical books/documents. Use when researching
  signals, runtime/ML/system issues, verifying concepts, or auditing features
  against the knowledge base.
---

# Orgonite Master Knowledge Base

The Orgonite Master KB is the same system referred to elsewhere as the Olson KB.
It is a local hybrid knowledge base containing ingested books and documents across
both esoteric and technical domains, and it is the authoritative local source for
what signals and engineering direction SHOULD be in the trading system.

## Location
`C:\Users\C\Desktop\MY GOOGLE DRIVE\Orgonite master`

## Query Methods (use Bash tool)

### Vector/Semantic Search (best for concepts)
```bash
cd "C:/Users/C/Desktop/MY GOOGLE DRIVE/Orgonite master" && python kb.py vsearch "gematria bitcoin number energy" --limit 10
```

### Full-text Search (best for exact terms)
```bash
cd "C:/Users/C/Desktop/MY GOOGLE DRIVE/Orgonite master" && python kb.py search "Schumann resonance" --limit 10
```

### Smart Search (hybrid — tries both, returns best)
```bash
cd "C:/Users/C/Desktop/MY GOOGLE DRIVE/Orgonite master" && python kb.py smart "planetary retrograde market cycles" --limit 10
```

### When semantic chunk recall is weak
Use direct keyword/title lookups against the ingested text to verify alignment with the source books:
```bash
cd "C:/Users/C/Desktop/MY GOOGLE DRIVE/Orgonite master" && python kb.py search "runtime OR deployment OR orchestration" --limit 10
```

## Key Signal Categories to Research
- **Gematria**: Bitcoin (33, 68, 213, 231, 312, 321, 132, 123), date gematria, headline gematria
- **Numerology**: Number energy (3, 6, 9, 11, 13, 33, 44), angel numbers, life path numbers
- **Astrology**: Planetary transits, Saturn/Jupiter cycles, retrograde periods, lunar phases
- **Kabbalah**: Sephiroth, Tree of Life correspondences, Hebrew letter values
- **Sacred Geometry**: Fibonacci, golden ratio in price patterns
- **Energy Medicine**: Schumann resonance, geomagnetic activity, solar cycles
- **Hermeticism**: As above so below — macro cycles reflecting in markets

## Feature Audit Protocol
When auditing features, for each signal category:
1. Query the KB for all relevant concepts: `python kb.py smart "<category>" --limit 20`
2. Cross-reference against `v3.3/feature_library.py`
3. Document what's present vs missing
4. NEVER suggest dropping features — only adding new ones
5. Every concept in the KB that has a quantifiable market signal belongs in feature_library.py

## Adding New Books
Drop PDFs in the `drop_here/` folder and run the ingest script.

## CRITICAL
The matrix thesis says MORE signals = stronger predictions. Every book in this KB
represents potential features. The audit goal is to ensure ZERO signal leakage.
