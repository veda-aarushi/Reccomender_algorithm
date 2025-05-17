import os

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
TRAIN_CSV  = os.getenv("TRAIN_CSV", "data/patagonia.csv")
DEFAULT_N  = int(os.getenv("DEFAULT_N", "5"))

# New autoencoder settings
MODEL_TYPE    = os.getenv("MODEL_TYPE", "tfidf")         # "tfidf" or "autoencoder"
LATENT_DIM    = int(os.getenv("LATENT_DIM", "64"))
AE_EPOCHS     = int(os.getenv("AE_EPOCHS", "10"))
AE_BATCH_SIZE = int(os.getenv("AE_BATCH_SIZE", "32"))
