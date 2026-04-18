"""
Logging and utilities
"""

import logging
import json
from pathlib import Path
from datetime import datetime


class TrainingLogger:
    """Training logging utility"""
    
    def __init__(self, log_dir: str = './logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger('wavDINO')
        handler = logging.FileHandler(self.log_dir / 'training.log')
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log(self, message: str, level: str = 'info'):
        """Log message"""
        if level == 'info':
            self.logger.info(message)
        elif level == 'warning':
            self.logger.warning(message)
        elif level == 'error':
            self.logger.error(message)
    
    def log_config(self, config: dict):
        """Log configuration"""
        config_file = self.log_dir / f'config_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)


class AverageMeter:
    """Compute and store the average and current value"""
    
    def __init__(self, name: str, fmt: str = ':.4f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


if __name__ == "__main__":
    logger = TrainingLogger()
    logger.log("Test log message")
    
    meter = AverageMeter('Loss')
    for i in range(10):
        meter.update(0.5 - i*0.01)
    print(meter)
