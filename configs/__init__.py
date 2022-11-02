from pathlib import Path
from utils.confort import select_cfg

Config = select_cfg(Path(__file__).parent)