"""
Auto-logging module - automatically saves all output to logs/ folder.

Usage:
    from utils.logger import setup_logging
    setup_logging("script_name")  # Call at the start of your script
    
    # Now all print() and errors go to both console AND log file
"""

import sys
import os
from datetime import datetime
from pathlib import Path

# PyTorch Lightning callback for clean epoch logging
try:
    from pytorch_lightning.callbacks import Callback
    
    class EpochLogger(Callback):
        """Custom callback to log epoch numbers (clean output for log files)."""
        def on_train_epoch_start(self, trainer, pl_module):
            print(f"   Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}", flush=True)
except ImportError:
    # PyTorch Lightning not available
    EpochLogger = None


class TeeOutput:
    """Duplicates output to both console and log file.
    
    Captures completed progress bars (100%) but filters intermediate updates.
    """
    
    def __init__(self, log_file, original_stream):
        self.log_file = log_file
        self.original = original_stream
        self.last_progress_line = ""
        
    def write(self, message):
        # Always write to console
        self.original.write(message)
        self.original.flush()
        
        # Handle progress bars specially
        if self._is_progress_update(message):
            # Store the latest progress line (might be the final one)
            # Clean it up - remove \r and keep content
            clean = message.replace('\r', '').strip()
            if clean:
                self.last_progress_line = clean
            return
        
        # If this is a newline after progress bar, log the completed progress
        if message == '\n' and self.last_progress_line:
            self.log_file.write(self.last_progress_line + '\n')
            self.log_file.flush()
            self.last_progress_line = ""
            return
        
        # Normal message - log if should
        if self._should_log(message):
            self.log_file.write(message)
            self.log_file.flush()
    
    def _is_progress_update(self, message: str) -> bool:
        """Check if this is a progress bar update."""
        # Progress bars use \r to overwrite and contain % or progress chars
        if '\r' in message:
            return True
        # Also check for tqdm-style patterns
        if any(c in message for c in ['█', '▏', '▎', '▍', '▌', '▋', '▊', '▉', '━', '#']):
            if '%' in message or 'it/s' in message or 'file/s' in message:
                return True
        return False
    
    def _should_log(self, message: str) -> bool:
        """Check if message should be written to log file."""
        # Always allow newlines
        if message == '\n' or message.endswith('\n'):
            return True
        
        # Skip ANSI escape sequences (cursor movement, colors, etc.)
        ansi_sequences = ['[A', '[B', '[C', '[D', '[K', '[2K', '[J', '[?25l', '[?25h']
        if any(seq in message for seq in ansi_sequences):
            return False
        
        # Skip if contains ESC character
        if '\x1b' in message:
            return False
        
        # Skip very short messages without newlines (likely fragments)
        if len(message.strip()) < 3:
            return False
        
        return True
        
    def flush(self):
        self.original.flush()
        self.log_file.flush()
        
    def isatty(self):
        return self.original.isatty()


def setup_logging(script_name: str) -> str:
    """
    Setup automatic logging to file.
    
    Args:
        script_name: Name for the log file (e.g., "train_models", "predict")
        
    Returns:
        Path to the log file
    """
    # Create logs directory in project root
    project_root = Path(__file__).parent.parent.parent
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{timestamp}_{script_name}.log"
    log_path = log_dir / log_filename
    
    # Open log file
    log_file = open(log_path, 'w', encoding='utf-8')
    
    # Write header
    log_file.write(f"{'='*60}\n")
    log_file.write(f"Log started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"Script: {script_name}\n")
    log_file.write(f"{'='*60}\n\n")
    log_file.flush()
    
    # Redirect stdout and stderr
    sys.stdout = TeeOutput(log_file, sys.stdout)
    sys.stderr = TeeOutput(log_file, sys.stderr)
    
    print(f"📝 Logging to: {log_path}")
    
    return str(log_path)
