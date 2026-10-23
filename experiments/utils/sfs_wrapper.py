"""
sfs_wrapper.py
──────────────
Thin wrapper around the gasol_optimizer SFS tool.
Exposes get_sfs, compare_blocks, and the raw helpers needed by env.py.
"""

import sys
import os
from pathlib import Path
from utils.logger import Logger

ROOT_DIR = Path(__file__).resolve().parents[1]
UTILS_DIR = ROOT_DIR / "utils"
GASOL_DIR = Path("/app/gasol-optimizer")

# Add both the package parent and the gasol_optimizer root so:
# 1) `import gasol_optimizer.gasol_asm` works
# 2) internal local imports like `import statistics` resolve to the local package
for module_path in (GASOL_DIR, UTILS_DIR):
    module_path_str = str(module_path)
    if module_path_str not in sys.path:
        sys.path.insert(0, module_path_str)

try:
    from gasol-optimizer.gasol_asm import (
        parse_blocks_from_plain_instructions,
        compute_original_sfs_with_simplifications,
        compare_asm_block_asm_format,
        extended_json_with_minlength,
        extended_json_with_instr_dep_and_bounds,
    )
    from global_params.options import OptimizationParams
except ImportError as e:
    raise ImportError(f"Could not import gasol_optimizer: {e}")


class SFSBridge:
    def __init__(self, logger:Logger=None):
        self.params = OptimizationParams()
        self.params.input_file = "rl_block.txt"
        self.logger = logger

    # ── Public helpers (used by env.py directly) ───────────────────────────

    def get_sfs(self, opcode_string:str, type="original", original_str=""):
        """
        Returns the SFS dict for *opcode_string*, or None if unparseable.
        """
        try:
            blocks = parse_blocks_from_plain_instructions(opcode_string)
            for block in blocks:
                sfs = compute_original_sfs_with_simplifications(
                    block, self.params
                )[0]["syrup_contract"]["isolated_block_0_0"]
            return sfs
        except Exception as e:
            if self.logger is not None:
                self.logger.log_generation_error(opcode_string, type, original_str, e)
            return None

    def get_extended_sfs(self, opcode_string:str):
        """
        Returns the complete SFS dict for *opcode_string*, or None if unparseable.
        """
        try:
            blocks = parse_blocks_from_plain_instructions(opcode_string)
            for block in blocks:
                sfs =  compute_original_sfs_with_simplifications(
                    block, self.params
                )[0]["syrup_contract"]["isolated_block_0_0"]
                sfs = extended_json_with_minlength(extended_json_with_instr_dep_and_bounds(sfs))
            return sfs
        except Exception as e:
            return None


    def compare_blocks(self, old_str, new_str):
        """Compare two opcode strings. Prints result."""
        try:
            old = parse_blocks_from_plain_instructions(old_str)[0]
            new = parse_blocks_from_plain_instructions(new_str)[0]
            b, s = compare_asm_block_asm_format(old, new, self.params)
            return b, s
        except Exception as e:
            if self.logger is not None:
                self.logger.log_comparison_error(old_str, new_str, e)
            return False, {"reason": f"compare_error: {e}"}



if __name__ == "__main__":
    bridge = SFSBridge()
    bridge.get_sfs("PUSH 2 PUSH 3 ADD")
    old = "POP PUSH f891ff6d7f793ee0bb75b22c39b418cfa3e45626 PUSH 4 PUSH0 PUSH 100 EXP DUP2 SLOAD DUP2 PUSH ffffffffffffffffffffffffffffffffffffffff MUL NOT AND SWAP1 DUP4 PUSH ffffffffffffffffffffffffffffffffffffffff AND MUL OR SWAP1 SSTORE POP PUSH 9 PUSH a PUSH [tag] 18 SWAP2 SWAP1 PUSH [tag] 2"
    new = "POP PUSH 12 PUSH 4 PUSH0 PUSH 100 EXP DUP2 SLOAD DUP2 PUSH ff MUL NOT AND SWAP1 DUP4 PUSH ff AND MUL OR SWAP1 SSTORE POP PUSH 9 PUSH a PUSH [tag] 7 SWAP2 SWAP1 PUSH [tag] 8"
    bridge.compare_blocks(old, new)
    bridge.compare_blocks(old, old)
