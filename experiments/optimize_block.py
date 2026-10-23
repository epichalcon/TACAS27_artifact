import argparse
import random 
import torch
import pathlib
import re
import os
import json

from pathlib import Path

from torch_geometric.data import Batch

from graph_2_seq.encoder import G2SEncoder
from graph_2_seq.decoder import G2SDecoder
from graph_2_seq.preprocess import json_to_pyg

from utils.sfs_wrapper import SFSBridge
from utils.notify import get_last_version

from utils.gasol_optimizer.ai_optimizer.sdg import Sdg

NODE_FEAT_DIM = 5
HIDDEN_DIM    = 128
SOS_ID        = 1
EOS_ID        = 2
MAX_LEN       = 128

def load_model(evaluate: str, checkpoint_path: str, vocab_size: int):
    """Load encoder + decoder from an checkpoint dict."""
    ckpt    = torch.load(checkpoint_path, weights_only=False)
    encoder = G2SEncoder(node_feat_dim=NODE_FEAT_DIM, hidden_dim=HIDDEN_DIM)
    decoder = G2SDecoder(vocab_size=vocab_size, hidden_dim=HIDDEN_DIM)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    encoder.eval()
    decoder.eval()
    epoch      = ckpt.get("epoch", "?")
    avg_reward = ckpt.get("avg_reward", float("nan"))
    equiv_rate = ckpt.get("equiv_rate", float("nan"))
    print(f"  Loaded {evaluate} checkpoint — epoch {epoch} | "
          f"avg reward {avg_reward:.3f} | "
          f"equiv rate {equiv_rate * 100:.1f}%")
    return encoder, decoder


def prepare_block(opcode_string: str, sfs: SFSBridge):
    orig_sfs = sfs.get_extended_sfs(opcode_string)
    if orig_sfs is None:
        print("  ❌ Unable to generate initial sfs")
        return None

    try:
        sdg = Sdg(orig_sfs)
        sdg = sdg.generate_json(filename=None)
    except Exception as e:
        print(f"  ⚠️  Comparison error: {e}")
        sdg = None
    
    return sdg
    

def load_dataset(dataset_path: str):
    container = torch.load(dataset_path, weights_only=False)
    dataset = container["dataset"]
    vocab = container["vocab"]
    constants = container.get("constants", [])
    return dataset, vocab, constants


def predict_sequence(encoder, decoder, data_sample, device):
    """
    Autoregressively generate a token sequence for one graph sample.
    Returns a list of token IDs (no SOS/EOS).
    """
    batch = Batch.from_data_list([data_sample]).to(device)

    with torch.no_grad():
        encoder_context, memory_mask = encoder(
            batch.x, batch.edge_index, batch.batch
        )

        generated    = torch.tensor([[SOS_ID]], dtype=torch.long, device=device)
        output_tokens = []

        for _ in range(MAX_LEN):
            logits     = decoder(generated, encoder_context,
                                 memory_key_padding_mask=memory_mask)
            next_token = logits[:, -1, :].argmax(dim=-1).item()

            if next_token == EOS_ID:
                break

            output_tokens.append(next_token)
            generated = torch.cat(
                [generated,
                 torch.tensor([[next_token]], dtype=torch.long, device=device)],
                dim=1,
            )

    return output_tokens


# ── Opcode helpers ──────────────────────────────────────────────────────────

def ids_to_str(token_ids, vocab_inv, skip_special=True, constants_map=None):
    parts = []
    for tid in token_ids:
        tok = vocab_inv.get(tid, f"ID:{tid}")
        if skip_special and tok in ("<PAD>", "<SOS>", "<EOS>"):
            continue
        if constants_map and tok in constants_map:
            tok = str(constants_map[tok])
        parts.append(tok)
    return " ".join(parts)


def sample_to_str(data_sample, vocab_inv, constants_map=None):
    """Convert a Data object's target_seq to an opcode string."""
    ids = [
        t.item() if hasattr(t, "item") else t
        for t in data_sample.target_seq
    ]
    return ids_to_str(ids, vocab_inv, constants_map=constants_map)
    

def calculate_gas(sfs):
    """Return the gas cost represented by an SFS dictionary.

    If the SFS already contains a trusted precomputed `current_cost`, use it.
    Otherwise, fall back to summing the `gas` values from each user instruction.
    """
    if sfs is None:
        raise ValueError("SFS data must not be None")

    if isinstance(sfs.get("current_cost", None), (int, float)):
        return sfs["current_cost"]

    user_instrs = sfs.get("user_instrs")
    if isinstance(user_instrs, list):
        return sum(int(instr.get("gas", 0)) for instr in user_instrs)

    raise ValueError("Unable to calculate gas from provided SFS structure")


def perform_evaluation(orig_len, pred_len, pred_str, sfs, original_str, idx=None):
    result = {
        "idx":       idx,
        "orig_len":  orig_len,
        "pred_len":  pred_len,
        "equiv":     False,
        "valid":     False,
        "gas_saved": 0,
        "length_saved": 0,
        "reason":    "empty",
        "original_gas": orig_len 
    }

    if not pred_str:
        print("  ⚠️  Empty prediction.")
        return result

    # Validate predicted sequence
    pred_sfs = sfs.get_sfs(pred_str)
    if pred_sfs is None:
        print("  ❌ Predicted sequence is INVALID bytecode.")
        result["reason"] = "invalid"
        return result

    result["valid"] = True

    print(f"\n{'─' * 70}")
    print(f"  Original  : {original_str}")
    print(f"  Predicted : {pred_str if pred_str else '(empty)'}")

    # Validate original (should always pass, but check defensively)
    orig_sfs = sfs.get_sfs(original_str)
    if orig_sfs is None:
        print("  ⚠️  Original sequence could not be parsed — skipping comparison.")
        result["reason"] = "original_invalid"
        return result

    # Compare
    try:
        equiv, reason = sfs.compare_blocks(original_str, pred_str)
    except Exception as e:
        print(f"  ⚠️  Comparison error: {e}")
        result["reason"] = f"compare_error: {e}"
        return result

    result["equiv"]  = equiv
    result["reason"] = reason

    result["orig_len"]  = orig_sfs["init_progr_len"]
    result["pred_len"] = pred_sfs["init_progr_len"]
    
    orig_gas = calculate_gas(orig_sfs)
    result["original_gas"] = orig_gas

    if equiv:
        gas_saved = max(0, orig_gas - calculate_gas(pred_sfs))
        result["gas_saved"] = gas_saved
        result["length_saved"] = result["orig_len"] - result["pred_len"]
        length_saved = result["length_saved"]
        if length_saved > 0:
            print(f"  ✅ EQUIVALENT — saved {length_saved} instr "
                  f"({length_saved / result["orig_len"] * 100:.1f}% reduction)")
        else:
            print(f"  ✅ EQUIVALENT — same length")
    else:
        print(f"  ❌ NOT equivalent — {reason}")
            
    result["pred_str"] = pred_str

    return result


def evaluate_block(encoder, decoder, sfs, block,
                    vocab, device, constants_map=None):
    """
    Generate a sequence for one sample, verify equivalence, and print results.

    Returns a result dict for aggregate statistics.
    """

    try:
        data = json_to_pyg(block, vocab)
    except Exception as e:
        print(f"Error while generating data: {e}")
        return {
            "orig_len": 0,
            "pred_len": 0,
            "equiv": False,
            "valid": False,
            "original_gas": 0,
            "gas_saved": 0,
            "length_saved": 0,
            "reason": "preprocess_error",
        }

    vocab_inv = {v: k for k, v in vocab.items()}
    original_str = block.get("original_seq", "")

    pred_ids = predict_sequence(encoder, decoder, data, device)
    pred_str = ids_to_str(pred_ids, vocab_inv, constants_map=constants_map)

    orig_len = len(original_str.split())
    pred_len = len(pred_str.split()) if pred_str else 0


    result = perform_evaluation(orig_len, pred_len, pred_str, sfs, original_str)

    
    if "PUSH values are different" in result["reason"]:
        orig_push_instr = {}
        pred_push_instr = {}

        orig_sfs = sfs.get_sfs(original_str)
        for instr in orig_sfs["user_instrs"]:
            if instr["push"]:
                if len(instr["disasm"].split()) > 1:
                    value = str(instr["value"][0])
                else:
                    value = hex(instr["value"][0]).replace("0x", "")
                orig_push_instr[instr["id"]] = (instr["outpt_sk"][0], value)

        pred_sfs = sfs.get_sfs(pred_str)
        for instr in pred_sfs["user_instrs"]:
            if instr["push"]:
                if len(instr["disasm"].split()) > 1:
                    value = str(instr["value"][0])
                else:
                    value = hex(instr["value"][0]).replace("0x", "")
                pred_push_instr[instr["id"]] = (instr["outpt_sk"][0], value)


        diff_push_values = [(p1[1][1], p2[1][1]) for p1, p2 in (zip(orig_push_instr.items(), pred_push_instr.items())) if p1 != p2]

        inv_constants = {k: v for v, k in constants_map.items()}
        for diff in diff_push_values:
            constants_map[inv_constants[diff[0]]] = diff[1]
            
        pred_str = ids_to_str(pred_ids, vocab_inv, constants_map=constants_map)

        orig_len = len(original_str.split())
        pred_len = len(pred_str.split()) if pred_str else 0

        result = perform_evaluation(orig_len, pred_len, pred_str, sfs, original_str)

    result["pred_str"] = pred_str
    return result

def main(args):
    # ── SFS bridge ─────────────────────────────────────────────────────
    print("\n🔧 Initialising SFS bridge ...")
    sfs = SFSBridge()

    with open(args.bl) as f:
        block_code = f.read()
    block = prepare_block(block_code, sfs)


    random.seed(args.seed)
    device = torch.device("cpu")


    # ── Load dataset ───────────────────────────────────────────────────
    print(f"📂 Loading dataset from {args.dataset} ...")
    container = torch.load(args.dataset, weights_only=False)
    dataset   = container["dataset"]
    vocab     = container["vocab"]
    print(f"   {len(dataset)} graphs | {len(vocab)} tokens")

    # ── Load model ─────────────────────────────────────────────────────
    if args.evaluate == "rl":
        print(f"\n🤖 Loading RL model from {args.rl_model} ...")
        encoder, decoder = load_model("RL", args.rl_model, len(vocab))
    else:
        print(f"\n🤖 Loading G2S model from {args.g2s_model} ...")
        encoder, decoder = load_model("G2S", args.g2s_model, len(vocab))

    # ── Evaluate ───────────────────────────────────────────────────────
    evaluate_block(encoder, decoder, sfs, block, vocab, device, block["constants"])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Model evaluation: ")
    parser.add_argument("--rl-model",  default=None,
                        help="Path to RL model version")
    parser.add_argument("--g2s-model",  default=None,
                        help="Path to graph to sequene model version")
    parser.add_argument("--dataset",     default=None,
                        help="Path to processed_dataset.pt")

    parser.add_argument("--n",    type=int, default=10,
                        help="Number of random samples to evaluate")
    parser.add_argument("--idx",  type=int, default=None,
                        help="Evaluate a specific sample index instead of random")
    parser.add_argument("--seed", type=int, default=42)


    parser.add_argument("--evaluate", choices=["rl", "g2s"], default="rl",
                        help="Path to RL model version")

    
    parser.add_argument("--bl", help="Block to Optimize")

    args = parser.parse_args()

    if args.rl_model is None:
        model = get_last_version("RLloop", "model")
        args.rl_model = pathlib.Path("RLloop", f"model{model}", "best_rl_model.pt")

    if args.dataset is None: 
        cache = get_last_version("graph_2_seq/caches", "cache")
        args.dataset = pathlib.Path("graph_2_seq", "caches", f"cache{cache}.pt")

    if args.g2s_model is None: 
        model = get_last_version("graph_2_seq", "model")
        args.g2s_model = pathlib.Path("graph_2_seq", f"model{model}", "best_model.pt")

    main(args)
