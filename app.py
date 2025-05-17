from flask import Flask, request, jsonify, render_template
from content_engine import ContentEngine
from redis.exceptions import RedisError

app = Flask(__name__, template_folder="templates")
engine = ContentEngine()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/train", methods=["POST"])
def train():
    data = request.get_json() or {}
    path = data.get("csv_path")
    if not path:
        return jsonify({"error": "csv_path required"}), 400

    try:
        engine.train(path)
        return jsonify({"status": "trained"}), 200
    except Exception as e:
        return jsonify({"error": f"Training failed: {e}"}), 500

@app.route("/recommend/<int:item_id>")
def recommend(item_id):
    # read count and fallback toggle from query params
    n = int(request.args.get("n", 5))
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
