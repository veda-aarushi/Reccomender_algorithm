import sys
from content_engine import ContentEngine

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train.py <data.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    engine = ContentEngine()
    print(f"Training on {csv_path}…")
    engine.train(csv_path)
    print("Done.")
