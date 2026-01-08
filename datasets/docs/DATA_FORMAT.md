# Data Format Documentation

## Sermon Data Format (Section 4.1)

### Chunk Record Schema (LOCKED)

Each sermon chunk must contain the following fields:

```python
{
    "chunk_id": str,          # Unique identifier (e.g., "47-0412--M_chunk_003")
    "date_id": str,           # Sermon identifier (e.g., "47-0412--M")
    "paragraph_start": int,   # Starting paragraph number
    "paragraph_end": int,     # Ending paragraph number
    "chunk_index": int,       # Position within sermon (for expansion)
    "text": str,              # The actual chunk text
}
```

### Date ID Format (Section 1.2 - LOCKED)

```
dd-mm-yy--<T>
```

Where `<T>` ∈ `{M, E}`:
- `M` = morning sermon
- `E` = evening sermon

Examples:
- `47-0412--M` = April 12, 1947, morning
- `64-1226--E` = December 26, 1964, evening

### Chunking Rules (Section 5.1 - LOCKED)

1. **Target size**: ~350 tokens per chunk
2. **Paragraph-aware**: Cut only at paragraph boundaries
3. **Long paragraphs**: If too long, split on sentence boundaries
4. **No overlap**: Retrieve multiple chunks and expand as needed
5. **Store ranges**: Always store paragraph_start and paragraph_end

### Training Data Format (Section 8.2)

For LoRA training, use JSONL format:

```jsonl
{"text": "...", "date_id": "47-0412--M", "language": "en", "paragraph_start": 1, "paragraph_end": 3}
{"text": "...", "date_id": "47-0412--M", "language": "en", "paragraph_start": 4, "paragraph_end": 6}
```

Required fields:
- `text`: The text content
- `date_id`: Sermon identifier
- `language`: Language code (ISO 639-1)
- `paragraph_start`: Starting paragraph
- `paragraph_end`: Ending paragraph

### Q/A Instruction Data Format (Section 8.3)

For instruction tuning:

```jsonl
{
  "question": "What did Brother Branham say about faith?",
  "answer": "According to [47-0412--M: ¶2–¶3], Brother Branham explained...",
  "context": "...",
  "date_id": "47-0412--M",
  "chunk_ids": ["47-0412--M_chunk_003", "47-0412--M_chunk_004"],
  "mode": "synthesis"
}
```

## File Organization

```
data/
├── raw/                    # Raw sermon files
│   ├── english/
│   ├── spanish/
│   └── ...
├── processed/              # Processed chunks
│   ├── chunks.jsonl
│   └── metadata.json
├── training/               # Training datasets
│   ├── continued_pretrain.jsonl
│   ├── instruction_tune.jsonl
│   └── splits/
│       ├── train.jsonl
│       ├── val.jsonl
│       └── test.jsonl
└── indices/                # Built indices
    ├── bm25.pkl
    └── faiss.index
```

## Important Notes

1. **No offsets**: We don't use character/token offsets, only paragraph ranges
2. **Deterministic sharding**: For reproducibility in training
3. **Canonical references**: date_id is the stable locator, not sermon title
4. **Deduplication**: Use chunk_id for deduplication in retrieval

