import pandas as pd
import redis
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class ContentEngine:
    SIMKEY = "p:smlr:%s"

    def __init__(self, redis_url="redis://localhost:6379/0"):
        # connect to Redis
        self._r = redis.from_url(redis_url)
        self.df = None
        self.last_csv = None

    def train(self, csv_path):
        # load & remember the dataframe
        df = pd.read_csv(csv_path)
        self.df = df
        self.last_csv = csv_path

        # build TF-IDF matrix over descriptions
        tf = TfidfVectorizer(ngram_range=(1, 3), stop_words="english")
        matrix = tf.fit_transform(df["description"])
        sims = linear_kernel(matrix, matrix)

        # clear any prior data
        self._r.flushdb()

        # for each item, store top-100 similarities
        for idx, row in df.iterrows():
            top_idxs = sims[idx].argsort()[:-101:-1]
            mapping = {
                str(int(df.loc[i, "id"])): float(sims[idx][i])
                for i in top_idxs if i != idx
            }
            self._r.zadd(self.SIMKEY % int(row["id"]), mapping)

    def predict(self, item_id, n=5, use_fallback=True):
        """
        Returns up to n recommendations for the given item_id.
        If use_fallback=True and fewer than n are in Redis,
        fill the rest from the original CSV ordering at score=0.
        """
        key = self.SIMKEY % int(item_id)
        raw = self._r.zrange(key, 0, n - 1, withscores=True, desc=True)
        results = [{"id": int(m.decode()), "score": s} for m, s in raw]

        # fallback to CSV order if needed
        if use_fallback and self.df is not None and len(results) < n:
            seen = {r["id"] for r in results}
            for other in self.df["id"]:
                if other == item_id or other in seen:
                    continue
                results.append({"id": int(other), "score": 0.0})
                if len(results) >= n:
                    break

        return results
