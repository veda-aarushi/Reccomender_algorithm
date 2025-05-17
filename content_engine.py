# content_engine.py

import pandas as pd
import redis
import config

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class ContentEngine:
    SIMKEY = "p:smlr:%s"

    def __init__(self, redis_url: str = config.REDIS_URL):
        self._r = redis.from_url(redis_url)
        self.df = None

    def train(self, csv_path: str):
        df = pd.read_csv(csv_path)
        self.df = df

        tfidf = TfidfVectorizer(ngram_range=(1, 3), stop_words="english")
        matrix = tfidf.fit_transform(df["description"])
        sims = linear_kernel(matrix, matrix)

        self._r.flushdb()
        for idx, row in df.iterrows():
            top_idxs = sims[idx].argsort()[:-101:-1]
            mapping = {
                str(int(df.loc[i, "id"])): float(sims[idx][i])
                for i in top_idxs if i != idx
            }
            self._r.zadd(self.SIMKEY % int(row["id"]), mapping)

    def predict(self, item_id: int, n: int = config.DEFAULT_N, use_fallback: bool = True):
        key = self.SIMKEY % int(item_id)
        raw = self._r.zrange(key, 0, n - 1, withscores=True, desc=True)
        results = [{"id": int(m.decode()), "score": s} for m, s in raw]

        if use_fallback and self.df is not None and len(results) < n:
            seen = {r["id"] for r in results}
            for other in self.df["id"]:
                if other == item_id or other in seen:
                    continue
                results.append({"id": int(other), "score": 0.0})
                if len(results) >= n:
                    break

        return results
