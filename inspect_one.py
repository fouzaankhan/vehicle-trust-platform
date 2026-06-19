# inspect_config.py

from src.utils.config_loader import load_config

cfg = load_config()

print(cfg)
print()
print("TARGET =", cfg["features"]["target"])