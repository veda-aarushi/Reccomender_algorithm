from flask import Flask, request, jsonify, render_template
from flask_httpauth import HTTPBasicAuth
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from prometheus_flask_exporter import PrometheusMetrics
from flask_restx import Api, Resource, fields, reqparse
from redis.exceptions import RedisError

import config
from content_engine import ContentEngine
from tasks import train_task

app = Flask(__name__, template_folder="templates")

# Expose Prometheus metrics on /metrics
metrics = PrometheusMetrics(app)

# Serve the UI at the root
@app.route("/")
def index():
    return render_template("index.html")

# Basic auth for /train
auth = HTTPBasicAuth()
USERS = {"admin": "secret"}  # TODO: secure properly

@auth.verify_password
def verify(username, password):
    return USERS.get(username) == password

# Rate limiting
limiter = Limiter(key_func=get_remote_address, default_limits=["100 per hour"])
limiter.init_app(app)

# Swagger / RESTX setup
api = Api(app, version="1.0", title="Recommender API", doc="/docs")

engine = ContentEngine()

### TRAIN NAMESPACE ###
ns_train = api.namespace("train", description="Training operations")
train_model = api.model(
    "TrainModel",
    {"csv_path": fields.String(required=False, description="Path to CSV file")}
)

@ns_train.route("/")
class Train(Resource):
    @api.expect(train_model)
    @api.response(202, "Training queued")
    @auth.login_required
    @limiter.limit("5 per minute")
    def post(self):
        data = api.payload or {}
        path = data.get("csv_path") or config.TRAIN_CSV
        job = train_task.delay(path)
        return {"status": "queued", "job_id": job.id}, 202

### RECOMMEND NAMESPACE ###
ns_rec = api.namespace("recommend", description="Recommendation operations")
recommendation = api.model(
    "Recommendation",
    {
        "id": fields.Integer(description="Recommended item ID"),
        "score": fields.Float(description="Similarity score")
    }
)

parser = reqparse.RequestParser()
parser.add_argument("n", type=int, default=config.DEFAULT_N, help="Number of recommendations")
parser.add_argument("fallback", type=bool, default=True, help="Enable fallback")

@ns_rec.route("/<int:item_id>")
class Recommend(Resource):
    @api.expect(parser)
    @api.marshal_list_with(recommendation)
    @limiter.limit("50 per minute")
    def get(self, item_id):
        args = parser.parse_args()
        results = engine.predict(item_id, args["n"], use_fallback=args["fallback"])
        return results, 200

if __name__ == "__main__":
    app.run(debug=True)
