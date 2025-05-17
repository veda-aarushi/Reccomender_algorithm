from flask import Flask, request, jsonify, render_template
from flask_httpauth import HTTPBasicAuth
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from redis.exceptions import RedisError
import config
from content_engine import ContentEngine
from tasks import train_task

app = Flask(__name__, template_folder="templates")
auth = HTTPBasicAuth()
limiter = Limiter(app, key_func=get_remote_address, default_limits=["100 per hour"])
engine = ContentEngine()

USERS = {"admin": "secret"}  # replace with real credentials

@auth.verify_password
def verify(username, password):
    return USERS.get(username) == password

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/train", methods=["POST"])
@auth.login_required
@limiter.limit("5 per minute")
def train():
    data = request.get_json() or {}
    path = data.get("csv_path") or config.TRAIN_CSV
    try:
        job = train_task.delay(path)
        return jsonify({"status": "queued", "job_id": job.id}), 202
    except Exception as e:
        return jsonify({"error": f"Queuing failed: {e}"}), 500

@app.route("/recommend/<item_id>")
@limiter.limit("50 per minute")
def recommend(item_id):
    if not item_id.isdigit() or int(item_id) <= 0:
        return jsonify({"error": "item_id must be a positive integer"}), 400
    item_id = int(item_id)

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
