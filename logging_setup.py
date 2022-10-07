import logging
import sys, os

def setup(log_file, level=logging.INFO):
    """Function for setup logger"""
    logDir = "logs"
    if not os.path.exists(logDir):
        os.makedirs(logDir)

    logging.addLevelName(logging.WARNING, 'WARN')
    logging.addLevelName(logging.CRITICAL, 'CRIT')

    handler_file = logging.FileHandler(os.path.join(logDir, log_file))
    formatter = PathTruncatingFormatter('%(asctime)s %(levelname)5.5s [%(pathname)30s:%(lineno)3d] %(message)s')
    handler_stream = logging.StreamHandler(sys.stdout)
    handler_file.setFormatter(formatter)
    handler_stream.setFormatter(formatter)

    logger = logging.getLogger()
    
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())

    logger.setLevel(level)
    logger.addHandler(handler_file)
    logger.addHandler(handler_stream)

    return logger

class PathTruncatingFormatter(logging.Formatter):
    def format(self, record):
        if 'pathname' in record.__dict__.keys():
            # get right 30 characters
            record.pathname = record.pathname[-30:]
        return super(PathTruncatingFormatter, self).format(record)
