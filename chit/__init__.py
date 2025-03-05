from .chit import *
from .utils import *

# import os
# import logging
# from datetime import datetime
# from logging.handlers import RotatingFileHandler

# LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")
# LOG_LEVEL_CONSOLE = os.getenv("LOG_LEVEL_CONSOLE", "INFO")
# LOG_LEVEL_FILE = os.getenv("LOG_LEVEL_FILE", "DEBUG")

# LOGGER = logging.getLogger(__name__)
# LOGGER.setLevel(LOG_LEVEL)

# FORMATTER = logging.Formatter(
#     "{asctime} - {levelname} - {message}",
#     style="{",
#     datefmt="%Y-%m-%d %H:%M:%S",
# )

# CONSOLE_HANDLER = logging.StreamHandler()
# CONSOLE_HANDLER.setLevel(LOG_LEVEL_CONSOLE)
# CONSOLE_HANDLER.setFormatter(FORMATTER)
# LOGGER.addHandler(CONSOLE_HANDLER)

# os.makedirs(".logs", exist_ok=True)

# FILE_HANDLER = RotatingFileHandler(
#     f".logs/{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.log",
#     maxBytes=1024 * 1024,
#     backupCount=3,
# )
# FILE_HANDLER.setLevel(LOG_LEVEL_FILE)
# FILE_HANDLER.setFormatter(FORMATTER)
# LOGGER.addHandler(FILE_HANDLER)

