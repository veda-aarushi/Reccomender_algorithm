from flask import Flask, request, jsonify, render_template
from content_engine import ContentEngine

app = Flask(__name__, template_folder="templates")
engine = ContentEngine()

@app.route("/")
def index():
    # render the form & results page
    return render_template("index.html")

@app.route("/train", methods=["POST"])
def train():
    data = request.get_json()
    path = data.get("csv_path")
    if not path:
        return jsonify({"error": "csv_path required"}), 400
    engine.train(path)
    return jsonify({"status": "trained"}), 200

@app.route("/recommend/<int:item_id>")
def recommend(item_id):
    n = int(request.args.get("n", 5))
    results = engine.predict(item_id, n)
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
