
from gensim.models import Word2Vec
from scipy import sparse

class QPrep:

    def __init__(self,questions):
        self.questions = questions
        self.word2vec = False

    def load_w2v(self,w2v_path):
        self.word2vec = Word2Vec.load(w2v_path)

    def encode(self,question):
        emb = []
        for w in question.tokens:
            try:
                emb.append(self.word2vec[w.lower()])
            except:
                emb.append(300 * [0])
        return sparse.csr_matrix(emb)

    def tokenize_questions(self):
        for question in self.questions:
            question.tokenize()

    def encode_questions(self,w2v_path):
        self.load_w2v(w2v_path)
        questions_encoded = []
        questions_indices = []
        index = 0
        for question in self.questions:
            encoded_question = self.encode(question)
            questions_encoded.append(encoded_question)
            questions_indices.append([index,encoded_question.shape[0]])
            index += encoded_question.shape[0]
        return sparse.vstack(questions_encoded), questions_indices

    def add_question(self,questions_encoded,question,w2v_path):
        if not self.word2vec:
            self.load_w2v(w2v_path)
        encoded_question = self.encode(question)
        question_index = [questions_encoded.shape[0],encoded_question.shape[0]]
        return sparse.vstack([questions_encoded,encoded_question]), question_index
