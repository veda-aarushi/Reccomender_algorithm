import os

# Redis connection URL
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Default CSV to train on
TRAIN_CSV = os.getenv("TRAIN_CSV", "data/patagonia.csv")

# Default number of recommendations
DEFAULT_N = int(os.getenv("DEFAULT_N", "5"))
