import pandas as pd
import redis
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class ContentEngine:
    SIMKEY = "p:smlr:%s"

    def __init__(self, redis_url="redis://localhost:6379/0"):
        self._r = redis.from_url(redis_url)

    def train(self, csv_path):
        # 1) Load your data
        df = pd.read_csv(csv_path)

        # 2) Build TF-IDF matrix
        tf = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            stop_words="english"
        )
        matrix = tf.fit_transform(df["description"])

        # 3) Compute cosine similarities
        sims = linear_kernel(matrix, matrix)

        # 4) Clear any old recommendations
        self._r.flushdb()

        # 5) For each item, pick top-100 and add to Redis
        for idx, row in df.iterrows():
            # get indices of top-100 similar items (including itself)
            top_idxs = sims[idx].argsort()[:-101:-1]

            # build a mapping {member: score}
            mapping = {}
            for i in top_idxs:
                if i == idx:
                    continue
                other_id = int(df.loc[i, "id"])
                score = float(sims[idx][i])
                # use string keys so redis returns bytes we can decode
                mapping[str(other_id)] = score

            # now do a single zadd with that mapping
            self._r.zadd(self.SIMKEY % int(row["id"]), mapping)

    def predict(self, item_id, n=5):
        key = self.SIMKEY % int(item_id)
        # get top-n (highest scores first)
        raw = self._r.zrange(key, 0, n - 1, withscores=True, desc=True)

        # raw is list of (member_bytes, score)
        results = []
        for member, score in raw:
            # decode bytes to text, then to int
            member_id = int(member.decode("utf-8"))
            results.append({"id": member_id, "score": score})

        return results
