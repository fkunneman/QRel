
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SoftCosine:
    """
    class to apply the SoftCosine similarity metric to score the similarity between any two strings,
    using cosine similarity and based on tfidf values and the semantic representation of the words 
    returns a similarity score
    """

    def __init__(self,d,tfidf):
        self.dict = d
        self.tfidf = tfidf

    def apply_model(self,q1,q2):
        """ 
        apply softcosine given the embeddings and word tokens of two questions
        Implementation by Thiago Castro Ferreira, formula is based on:
        Sidorov, G., Gelbukh, A., Gómez-Adorno, H., & Pinto, D. (2014). 
        Soft similarity and soft cosine measure: Similarity of features in vector space model. 
        Computación y Sistemas, 18(3), 491-504.
        """

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

        # return the tfidf vectors for the two questions
        q1tfidf = self.tfidf[self.dict.doc2bow(q1.tokens)]
        q2tfidf = self.tfidf[self.dict.doc2bow(q2.tokens)]

        # intermediate steps for softcosine calculation
        q1q1 = np.sqrt(dot(q1tfidf, q1.emb, q1tfidf, q1.emb))
        q2q2 = np.sqrt(dot(q2tfidf, q2.emb, q2tfidf, q2.emb))

        # calculate softcosine
        softcosine = dot(q1tfidf, q1.emb, q2tfidf, q2.emb) / (q1q1 * q2q2)
        return softcosine
