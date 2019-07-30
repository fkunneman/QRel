
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SoftCosine:

    def __init__(self,d,tfidf):
        self.dict = d
        self.tfidf = tfidf

    def apply_model(self,q1,q2):

        def dot(q1tfidf, q1emb, q2tfidf, q2emb):
            cos = 0.0
            for i, w1 in enumerate(q1tfidf):
                for j, w2 in enumerate(q2tfidf):
                    if w1[0] == w2[0]:
                        cos += (w1[1] * w2[1])
                    else:
                        m_ij = max(0, cosine_similarity([q1emb[i]], [q2emb[j]])[0][0])**2
                        cos += (w1[1] * m_ij * w2[1])
            return cos

        q1tfidf = self.tfidf[self.dict.doc2bow(q1.tokens)]
        q2tfidf = self.tfidf[self.dict.doc2bow(q2.tokens)]

        q1q1 = np.sqrt(dot(q1tfidf, q1.emb, q1tfidf, q1.emb))
        q2q2 = np.sqrt(dot(q2tfidf, q2.emb, q2tfidf, q2.emb))

        softcosine = dot(q1tfidf, q1.emb, q2tfidf, q2.emb) / (q1q1 * q2q2)
        return softcosine
