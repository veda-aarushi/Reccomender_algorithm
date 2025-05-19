# Recommender Algorithm Service

A production-ready, content-based recommendation engine with optional collaborative-filtering fallback, exposed via a Flask REST API. Built for high performance and easy integration, featuring asynchronous training, caching, rate limiting, and monitoring.

![CI](https://img.shields.io/badge/build-passing-brightgreen) ![Python](https://img.shields.io/badge/python-3.9%2B-blue) ![Redis](https://img.shields.io/badge/redis-6%2B-orange)

---

## 📖 Overview

This service analyzes product descriptions to suggest similar items, solving cold-start scenarios with a TF-IDF–based content engine and falling back to collaborative filtering when needed.

Key points:

* **Content-Based**: TF-IDF vectorization of product descriptions
* **Collaborative Fallback**: CF on user-item interactions for sparse cases
* **Asynchronous Training**: Celery tasks queued for retraining
* **High-Performance Serving**: Redis for similarity storage & caching
* **REST API**: Flask + Flask-RESTX with interactive Swagger docs
* **Security & Quotas**: HTTP Basic Auth on training, rate limiting on all endpoints
* **Observability**: Prometheus metrics exposed via `/metrics`

---

## 🚀 Features

1. **Asynchronous Model Training**

   ```http
   POST /train
   ```

   * Protected by Basic Auth (`admin:secret`)
   * Rate-limited to 5 requests/minute
   * Returns a Celery job ID for status tracking

2. **Get Recommendations**

   ```http
   GET /recommend/<item_id>?n=5&fallback=true
   ```

   * `item_id`: ID of the anchor product
   * `n` (optional): number of recommendations (default: 5)
   * `fallback` (optional): enable CF fallback (default: true)
   * Rate-limited to 50 requests/minute

3. **Interactive API Docs**
   Visit `http://localhost:5000/docs` for Swagger UI and request examples.

4. **Metrics & Monitoring**
   ScrapePrometheus metrics at `http://localhost:5000/metrics` for request counts, latencies, and queue depths.

---

## 🛠️ Tech Stack

| Component          | Technology                |
| ------------------ | ------------------------- |
| Web Framework      | Flask + Flask-RESTX       |
| Asynchronous Tasks | Celery + Redis            |
| Similarity Store   | Redis Sorted Sets         |
| Vectorization & ML | scikit-learn TF-IDF       |
| Rate Limiting      | Flask-Limiter             |
| Auth               | Flask-HTTPAuth            |
| Caching            | Flask-Caching             |
| Monitoring         | prometheus-flask-exporter |

---

## ⚙️ Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/veda-aarushi/Recommender_algorithm.git
   cd Recommender_algorithm
   ```

2. **Create & activate virtualenv**

   ```bash
   python -m venv venv
   source venv/bin/activate    # macOS/Linux
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables**

   ```bash
   export REDIS_URL=redis://localhost:6379/0
   export TRAIN_CSV=data/patagonia.csv
   export DEFAULT_N=5
   ```

   On Windows PowerShell:

   ```powershell
   $Env:REDIS_URL = "redis://localhost:6379/0"
   $Env:TRAIN_CSV  = "data/patagonia.csv"
   $Env:DEFAULT_N  = "5"
   ```

5. **Start Redis** (if not already running):

   ```bash
   docker run -d --name redis -p 6379:6379 redis:6-alpine
   ```

---

## 🔧 Running the Service

1. **Start the Flask API**

   ```bash
   python app.py
   ```

   The API will run at `http://127.0.0.1:5000`.

2. **Start the Celery worker**

   ```bash
   celery -A celery_app.celery worker --pool=solo -Q train --loglevel=info
   ```

3. **Train the model**

   ```bash
   curl -u admin:secret -X POST http://localhost:5000/train \
     -H "Content-Type: application/json" \
     -d '{"csv_path":"data/patagonia.csv"}'
   ```

4. **Request recommendations**

   ```bash
   curl "http://localhost:5000/recommend/4?n=5&fallback=true"
   ```

---

## 📂 Project Structure

```
Recommender_algorithm/
├── app.py               # Flask app and API routes
├── celery_app.py        # Celery & Redis configuration
├── config.py            # Env-based constants
├── content_engine.py    # TF-IDF & CF logic
├── tasks.py             # Celery task definitions
├── data/                # Sample CSV data
│   └── patagonia.csv    # Patagonia product descriptions
├── requirements.txt     # Python dependencies
├── templates/           # HTML templates (Swagger UI)
├── static/              # CSS/JS assets
├── README.md            # This file
└── LICENSE              # MIT License
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "feat: add awesome feature"`
4. Push to your branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the [MIT License](LICENSE). Enjoy and build upon it freely!
