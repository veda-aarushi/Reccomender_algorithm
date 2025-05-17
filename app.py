from flask import Flask, request, jsonify, render_template
from redis.exceptions import RedisError
import config
from content_engine import ContentEngine

app = Flask(__name__, template_folder="templates")
engine = ContentEngine()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/train", methods=["POST"])
def train():
    data = request.get_json() or {}
    path = data.get("csv_path") or config.TRAIN_CSV

    try:
        engine.train(path)
        return jsonify({"status": "trained", "csv": path}), 200
    except FileNotFoundError:
        return jsonify({"error": f"CSV file not found: {path}"}), 400
    except Exception as e:
        return jsonify({"error": f"Training failed: {e}"}), 500

@app.route("/recommend/<item_id>")
def recommend(item_id):
    # Validate item_id
    if not item_id.isdigit() or int(item_id) <= 0:
        return jsonify({"error": "item_id must be a positive integer"}), 400
    item_id = int(item_id)

    # Validate n
    n_param = request.args.get("n", str(config.DEFAULT_N))
    if not n_param.isdigit() or int(n_param) <= 0:
        return jsonify({"error": "n must be a positive integer"}), 400
    n = int(n_param)

    use_fb = request.args.get("fallback", "true").lower() == "true"

    try:
        results = engine.predict(item_id, n, use_fallback=use_fb)
        return jsonify(results), 200
    except RedisError as re:
        return jsonify({"error": f"Redis error: {re}"}), 500
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
