"""
preprocess.py
─────────────
Reads all block JSON files from a directory, converts each one into a
PyTorch Geometric Data object, and saves the resulting list to a cache file.

Vocabulary
──────────
Instead of a hand-written dict we derive token IDs from the EVM opcode hex
codes at runtime.  The mapping is built once in `build_vocab()` and reused
for every file.  Three special tokens are prepended:

    <PAD>  →  0
    <SOS>  →  1
    <EOS>  →  2

Every opcode label is looked up via `opcode_to_hex()`.  If a label is not
found it falls back to the label string
itself so the vocabulary stays consistent.

Node features
─────────────
Each node carries a 5-element feature vector as stored in the JSON.  String
values (e.g. the operand field "0", "1", …) are cast to float.
"""

import os
import json
import torch
from torch_geometric.data import Data
from typing import Dict, List

from sklearn.model_selection import train_test_split

# ── Vocabulary ─────────────────────────────────────────────────────────────

_SPECIAL = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}


def build_vocab(json_dir: str) -> Dict[str, int]:
    """
    Scan all JSON files in *json_dir* and collect every unique opcode label
    that appears in any node whose type is OPCODE, plus every token that
    appears in the ``original_seq`` field.

    Token identity is the hex string returned by ``opcode_to_hex()``.
    Special tokens (<PAD>, <SOS>, <EOS>) occupy IDs 0-2.
    """
    tokens = set()

    for i, fname in enumerate(os.listdir(json_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(json_dir, fname)) as f:
            try:
                block = json.load(f)
            except json.JSONDecodeError:
                continue

        for k in block.get("constants", {}).keys():
            tokens.add(k)

        # Collect from node labels
        for node in block.get("nodes", []):
            if node.get("type") == "OPCODE":
                tokens.add(node["label"])

        for tok in block.get("seq_with_constants").split():
            if tok.startswith("DUP") or tok.startswith("SWAP") or tok == "POP":
                tokens.add(tok)

    vocab = dict(_SPECIAL)
    for idx, tok in enumerate(sorted(tokens), start=len(_SPECIAL)):
        vocab[tok] = idx
    return vocab


# ── Feature helpers ────────────────────────────────────────────────────────

def _node_features(node: dict) -> List[float]:
    """Cast the 5-element feature list to floats (strings become floats)."""
    feats = node.get("features", [0, 0, 0, 0, 0])
    result = []
    for v in feats:
        try:
            result.append(float(v))
        except (ValueError, TypeError):
            result.append(0.0)
    return result


def _tokenize_seq_with_constants(seq: str, vocab: Dict[str, int]) -> List[str]:
    """Split a sequence into tokens while preserving multi-word vocabulary tokens."""
    words = seq.split()
    tokens: List[str] = []
    multi_word_labels = [label.split() for label in vocab.keys() if " " in label]
    multi_word_labels.sort(key=len, reverse=True)

    i = 0
    while i < len(words):
        matched = False
        for label_words in multi_word_labels:
            length = len(label_words)
            if words[i:i + length] == label_words:
                tokens.append(" ".join(label_words))
                i += length
                matched = True
                break
        if not matched:
            tokens.append(words[i])
            i += 1
    return tokens


# ── Per-file conversion ────────────────────────────────────────────────────

def json_to_pyg(block: dict, vocab: Dict[str, int]) -> Data:
    """
    Convert a single block dict into a PyG Data object.

    Fields
    ──────
    x           : FloatTensor  [N, 5]          node feature matrix
    edge_index  : LongTensor   [2, E]           directed edge list
    target_seq  : LongTensor   [T]              token IDs incl. SOS & EOS
    """
    # ── Node features ──────────────────────────────────────────────────
    id_to_idx = {n["id"]: i for i, n in enumerate(block["nodes"])}
    x = torch.tensor(
        [_node_features(n) for n in block["nodes"]], dtype=torch.float
    )

    # ── Edges ──────────────────────────────────────────────────────────
    src_list, dst_list = [], []
    for edge in block.get("edges", []):
        s = id_to_idx.get(edge["src"])
        d = id_to_idx.get(edge["dst"])
        if s is not None and d is not None:
            src_list.append(s)
            dst_list.append(d)

    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    # ── Target sequence ────────────────────────────────────────────────
    pad_id = vocab["<PAD>"]
    sos_id = vocab["<SOS>"]
    eos_id = vocab["<EOS>"]

    tokens = []
    for token in _tokenize_seq_with_constants(block.get("seq_with_constants", ""), vocab):
        tokens.append(vocab.get(token, pad_id))

    target_seq_tensor = torch.tensor([sos_id] + tokens + [eos_id], dtype=torch.long)

    # Store as a plain Python list so PyG never auto-batches it
    # (PyG concatenates tensor attributes into a flat 1-D tensor in Batch)
    return Data(x=x, edge_index=edge_index, target_seq=target_seq_tensor.tolist())


# ── Folder-level entry point ───────────────────────────────────────────────

def _cache_paths(cache_file: str) -> Dict[str, str]:
    base_path, _ = os.path.splitext(cache_file)
    return {
        "vocab": f"{base_path}_vocab.pt",
        "train": f"{base_path}_train.pt",
        "val": f"{base_path}_val.pt",
    }


def preprocess_folder(json_dir: str, cache_file: str, val_split: float = 0.1) -> Dict[str, str]:
    """
    Process every .json file in *json_dir*, build a shared vocabulary, convert
    each block to a PyG Data object, and save the result in three caches:

    - vocab cache: vocabulary mapping
    - train cache: training graphs and constants
    - val cache: validation graphs and constants
    """
    print(f"Building vocabulary from {json_dir} …")
    vocab = build_vocab(json_dir)
    print(f"  → Vocabulary size: {len(vocab)} tokens")

    dataset: List[Data] = []
    constants_by_sample = []
    filenames = []
    skipped = 0
    files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
    for fname in sorted(files):
        path = os.path.join(json_dir, fname)
        try:
            with open(path) as f:
                block = json.load(f)
            data = json_to_pyg( block, vocab)
            # Skip degenerate graphs (no nodes or no sequence)
            if data.x.shape[0] == 0 or len(data.target_seq) <= 2:
                skipped += 1
                continue
            dataset.append(data)
            constants_by_sample.append(block.get("constants", {}))
            filenames.append(fname) 
        except Exception as e:
            print(f"  ⚠  Skipping {fname}: {e}")
            skipped += 1

    print(f"  → Processed {len(dataset)} blocks  ({skipped} skipped)")


    if len(dataset) <= 1:
        train_dataset, val_dataset = dataset, []
        train_constants, val_constants = constants_by_sample, []
        train_fnames, val_fnames = filenames, []
    else:
        (
            train_dataset,
            val_dataset,
            train_constants,
            val_constants,
            train_fnames,
            val_fnames,
        ) = train_test_split(
            dataset,
            constants_by_sample,
            filenames,
            test_size=val_split,
            random_state=42,  
            shuffle=True,
        )

    cache_paths = _cache_paths(cache_file)
    cache_dir = os.path.dirname(cache_file)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    torch.save({"vocab": vocab}, cache_paths["vocab"])
    torch.save({"dataset": train_dataset, "constants": train_constants, "fnames": train_fnames}, cache_paths["train"])
    torch.save({"dataset": val_dataset, "constants": val_constants, "fnames": val_fnames}, cache_paths["val"])

    print(f"  → Saved vocab cache to {cache_paths['vocab']}")
    print(f"  → Saved train cache to {cache_paths['train']}")
    print(f"  → Saved validation cache to {cache_paths['val']}")
    return cache_paths



if __name__ == "__main__":
    json_dir = "../intermediate/sdgs/0x/sdgs"
    vocab = build_vocab(json_dir)
    print(vocab)

    total = 0
    count = 0
    files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
    for fname in sorted(files):
        path = os.path.join(json_dir, fname)
        try:
            with open(path) as f:
                block = json.load(f)
            data = json_to_pyg(block, vocab)
            tg_seq = data.target_seq
            if 0 in tg_seq:
                count += 1
            total += 1
        except Exception as e:
            print(f"  ⚠  Skipping {fname}: {e}")
    print(f"{count}/{total}")