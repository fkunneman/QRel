
import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class TRLM:

    def __init__(self,d):
        self.dict = d
        self.alpha = False
        self.sigma = False
        self.prob_w_C = False

    def init_model(self,questions):
        tokens = sum([q.tokens for q in questions],[])
        Q_len = float(len(tokens))
        aux_w_Q = self.dict.doc2bow(tokens)
        aux_w_Q = dict([(self.dict[w[0]], (w[1]+1.0)/(Q_len+len(self.dict))) for w in aux_w_Q])

        w_Q = {}
        for w in aux_w_Q:
            if w[0] not in w_Q:
                w_Q[w[0]] = {}
            w_Q[w[0]][w] = aux_w_Q[w]

        self.alpha = 0.3
        self.sigma = 0.7
        self.prob_w_C = w_Q

    def save_model(self,trlm_path):
    	trlm_output = {'alpha':self.alpha,'sigma':self.sigma,'w_Q':self.prob_w_C}
    	with open(trlm_path,'w',encoding='utf-8') as output_file:
    		json.dump(trlm_output,output_file)

    def load_model(self,trlm_path):
        with open(trlm_path,'r',encoding='utf-8') as trlm_in:
        	translation = json.loads(trlm_in.read())
        self.alpha = translation['alpha']
        self.sigma = translation['sigma']
        self.prob_w_C = translation['w_Q']

    def apply_model(self,q1,q2):
       
        score = 0.0
        if len(q1.tokens) == 0 or len(q2.tokens) == 0: return 0.0

        Q = pd.Series(q2.tokens)
        Q_count = Q.count()

        t_Qs = []
        for t in q2.tokens:
            t_Q = float(Q[Q == t].count()) / Q_count
            t_Qs.append(t_Q)

        for i, w in enumerate(q1.tokens):
            try:
                w_C = self.prob_w_C[w[0]][w]
            except:
                w_C = 1.0 / len(self.dict)

            ml_w_Q = float(Q[Q == w].count()) / Q_count
            mx_w_Q = 0.0

            for j, t in enumerate(q2.tokens):
                w_t = max(0, cosine_similarity([q1.emb[i]], [q2.emb[j]])[0][0]) ** 2

                t_Q = t_Qs[j]
                mx_w_Q += (w_t * t_Q)
            w_Q = (self.sigma * mx_w_Q) + ((1-self.sigma) * ml_w_Q)
            score += np.log(((1-self.alpha) * w_Q) + (self.alpha * w_C))

        return score
