
from gensim.models import Word2Vec
from scipy import sparse

class QPrep:

    def __init__(self,questions):
        self.questions = questions
        self.word2vec = False

    def load_w2v(self,w2v_path):
        self.word2vec = Word2Vec.load(w2v_path)

    def tokenize_questions(self):
        for question in self.questions:
            question.tokenize()
