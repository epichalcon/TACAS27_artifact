import logging
import sys
from pathlib import Path


class NoErrorFilter(logging.Filter):
    """Filters out WARNING and higher log levels to keep logs clean."""
    def filter(self, record):
        return record.levelno < logging.WARNING


class Logger:
    def __init__(self, log_path:str = None, console_output: bool = True):
        if log_path is not None:
            self.log_dir = Path(log_path)
        else:
            self.log_dir = Path("logs")

        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize core logger
        self.logger = logging.getLogger("Logger")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False  # Prevent duplicate logging to root
        
        # Prevent adding duplicate handlers if class is re-instantiated
        if not self.logger.handlers:
            self._setup_handlers(console_output)

    def _setup_handlers(self, console_output: bool):
        # Formatters
        metric_formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        error_formatter = logging.Formatter('%(asctime)s - %(levelname)s -\n%(message)s\n' + '='*50)
        
        # 1. Training Metric Handler (Info only)
        train_file = self.log_dir / "training.log"
        train_handler = logging.FileHandler(train_file, mode="w")
        train_handler.setLevel(logging.INFO)
        train_handler.setFormatter(metric_formatter)
        train_handler.addFilter(NoErrorFilter())
        self.logger.addHandler(train_handler)
        
        # 2. Error/Exception Handler (Warning & Error only)
        error_file = self.log_dir / "errors.log"
        error_handler = logging.FileHandler(error_file, mode="w")
        error_handler.setLevel(logging.WARNING)
        error_handler.setFormatter(error_formatter)
        self.logger.addHandler(error_handler)
        
        # 3. Optional Stream Handler for real-time Terminal monitoring
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(metric_formatter)
            console_handler.addFilter(NoErrorFilter())
            self.logger.addHandler(console_handler)

    def log_metrics(self, epoch: int, max_epochs: int, avg_reward: float, equiv: float, pg: float, val: float, ent: float, kl: float):
        """Standardized helper for your RL epoch steps."""
        metric_msg = (
            f"Epoch {epoch:4d}/{max_epochs} | "
            f"Avg Reward: {avg_reward:7.3f} | "
            f"Equiv: {equiv:5.1f}% | "
            f"PG: {pg:7.4f} | "
            f"Val: {val:7.4f} | "
            f"Ent: {ent:7.4f} | "
            f"KL: {kl:7.4f}"
        )
        self.logger.info(metric_msg)


    def log_kl_invalid_metrics(self, epoch: int, max_epochs: int, avg_reward: float, equiv: float):
        """Standardized helper for your RL epoch steps."""
        metric_msg = (
            f"Epoch {epoch:4d}/{max_epochs} | "
            f"Avg Reward: {avg_reward:7.3f} | "
            f"Equiv: {equiv:5.1f}% | "
            f"[PPO update skipped — KL too high]"
        )
        self.logger.info(metric_msg)

    def log_info(self, msg: str):
        """For general non-error pipeline steps (e.g., saving checkpoints)."""
        self.logger.info(msg)

    def log_generation_error(self, block_bytecode: str, type:str, original_str, error_exception: Exception):
        """Dedicated helper to format and dump EVM generation errors cleanly."""
        error_msg = (
            f"An error occurred while processing the sfs of a {type} block:\n"
            f"block: {block_bytecode}\n"
            f"Error: {error_exception}\n"
            f"Original block: {original_str}\n"
        )
        self.logger.error(error_msg)


    def log_comparison_error(self, old_block: str, new_block: str, error_exception: Exception):
        error_msg = (
            f"An error occurred while comparing: \n"
            f"old block: {old_block}\n"
            f"new block: {new_block}\n"
            f"Error: {error_exception}\n"
        )
        self.logger.error(error_msg)

    def log_separate(self):
        self.logger.info("─" * 62)


    def log_g2s_epoch(self, epoch, train_loss, train_ter, val_loss, val_ter, current_lr):
        msg = (
            f"\n{epoch:>5} | "
            f"{train_loss:>11.4f} | "
            f"{train_ter * 100:>9.1f}% | "
            f"{val_loss:>10.4f} | " 
            f"{val_ter * 100:>8.1f}% | " 
            f"{current_lr:>10.2e}")

        self.logger.info(msg)



    def log_g2s_batch(self, desc, i, total_batches, loss, ter):
        msg = (
            f"  [{desc}] Batch {i:>4}/{total_batches} "
            f"| Loss {loss:.4f} "
            f"| TER {ter * 100:.1f}%")

        self.logger.info(msg)