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
        # Track progress bars by their prefix (e.g., "Epoch 0", "Validation DataLoader 0")
        self.progress_bars = {}
        
    def write(self, message):
        # Always write to console
        self.original.write(message)
        self.original.flush()
        
        # Handle progress bars specially
        if self._is_progress_update(message):
            # Extract progress bar prefix and percentage
            clean = message.replace('\r', '').strip()
            if clean:
                prefix = self._get_progress_prefix(clean)
                percent = self._get_progress_percent(clean)
                # Store the latest state of this progress bar
                self.progress_bars[prefix] = {'line': clean, 'percent': percent}
            return
        
        # If this is a newline after progress bars, check for completed ones
        if message == '\n':
            # Log any progress bars that reached 100%
            for prefix, data in list(self.progress_bars.items()):
                if data['percent'] >= 100:
                    self.log_file.write(data['line'] + '\n')
                    self.log_file.flush()
            self.progress_bars.clear()
            return
        
        # Normal message - log if should
        if self._should_log(message):
            self.log_file.write(message)
            self.log_file.flush()
    
    def _get_progress_prefix(self, message: str) -> str:
        """Extract the prefix/label of a progress bar."""
        # Format: "Label: X%|..." or "Label: |..."
        if ':' in message:
            return message.split(':')[0].strip()
        return message[:20]  # Fallback: first 20 chars
    
    def _get_progress_percent(self, message: str) -> float:
        """Extract percentage from progress bar message."""
        import re
        # Look for patterns like "100%", " 50%", etc.
        match = re.search(r'(\d+)%', message)
        if match:
            return float(match.group(1))
        # Check if it's a completed bar by looking at N/M pattern where N==M
        match = re.search(r'\|\s*(\d+)/(\d+)\s*\[', message)
        if match:
            current, total = int(match.group(1)), int(match.group(2))
            if total > 0:
                return (current / total) * 100
        return 0.0
    
    def _is_progress_update(self, message: str) -> bool:
        """Check if this is a progress bar update."""
        # Progress bars use \r to overwrite and contain % or progress chars
        if '\r' in message:
            return True
        # Check for tqdm-style progress bar characters
        if any(c in message for c in ['█', '▏', '▎', '▍', '▌', '▋', '▊', '▉', '━', '#']):
            if '%' in message or 'it/s' in message or 'file/s' in message:
                return True
        # Check for PyTorch Lightning / tqdm progress patterns
        # Patterns like: "Validation: |          | 0/? [00:00<?, ?it/s]"
        # or "Epoch 0:   0%|          | 38555/16195624 [02:57<20:38:23, 217.45it/s"
        if 'it/s' in message or 'it/s]' in message:
            return True
        if '|' in message and ('[' in message or '%' in message):
            # Likely a progress bar with format: "label: X%|bar| N/M [time]"
            return True
        if 'DataLoader' in message and ('/' in message or '%' in message):
            return True
        # Pattern: "Training: |" or "Validation: |" - progress bar start
        if message.strip().endswith('|') or '| 0/?' in message:
            return True
        return False
    
    def _should_log(self, message: str) -> bool:
        """Check if message should be written to log file."""
        # Skip ANSI escape sequences (cursor movement, colors, etc.)
        ansi_sequences = ['[A', '[B', '[C', '[D', '[K', '[2K', '[J', '[?25l', '[?25h']
        if any(seq in message for seq in ansi_sequences):
            return False
        
        # Skip if contains ESC character
        if '\x1b' in message:
            return False
        
        # Skip progress bar patterns that might have slipped through
        if 'it/s' in message or 'it/s]' in message:
            return False
        if '|' in message and '%|' in message:
            return False
        if 'DataLoader' in message and '/' in message and '[' in message:
            return False
        
        # Always allow newlines
        if message == '\n' or message.endswith('\n'):
            return True
        
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
