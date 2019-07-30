
import pickle

from gensim.summarization import bm25

class GV_BM25:

    def __init__(self):
        self.model = False

    def load_model(self,modelpath):
        with open(modelpath,'rb') as input_file:
            self.model = pickle.load(input_file)

    def init_model(self,questions):
        self.model = bm25.BM25(questions)

    def update_model(self,question):
        pass

    def save_model(self,modelpath):
        with open(modelpath, 'wb') as fid:
            pickle.dump(self.model, fid)
    
    def return_scores(self,question):
        return self.model.get_scores(question)

    def return_score(self,q1,q2_idx):
        return self.model.get_score(q1.tokens,q2_idx)
