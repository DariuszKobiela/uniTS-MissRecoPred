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


class TeeOutput:
    """Duplicates output to both console and log file.
    
    Filters out progress bars from log file (they use \\r for line overwrites).
    """
    
    def __init__(self, log_file, original_stream):
        self.log_file = log_file
        self.original = original_stream
        
    def write(self, message):
        # Always write to console
        self.original.write(message)
        self.original.flush()
        
        # Filter progress bars from log file
        # Progress bars use \r (carriage return) to overwrite lines
        # Also filter common progress bar patterns
        if self._should_log(message):
            self.log_file.write(message)
            self.log_file.flush()
    
    def _should_log(self, message: str) -> bool:
        """Check if message should be written to log file."""
        # Always log messages with newlines (real output, not progress bar updates)
        if '\n' in message:
            return True
        
        # Skip if contains carriage return without newline (progress bar overwrites)
        if '\r' in message:
            return False
        
        # Skip very short messages (likely progress bar fragments)
        if len(message.strip()) < 3:
            return False
        
        # Skip progress bar characters
        progress_chars = ['━', '█', '▏', '▎', '▍', '▌', '▋', '▊', '▉', '|']
        if any(char in message for char in progress_chars):
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
