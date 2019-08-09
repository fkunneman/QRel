
import pickle as p

from gensim.summarization import bm25

class GV_BM25:
    """
    class to train and apply BM25 to score the similarity between any two strings based on tfidf values of word tokens 
    returns a similarity score
    """

    def __init__(self):
        self.model = False

    def init_model(self,questions):
        # initialize BM25 model by reading in questions; the gensim implementation of BM25 is used (https://radimrehurek.com/gensim/summarization/bm25.html)
        self.model = bm25.BM25(questions)
    
    def return_scores(self,questiontokens):
        # return BM25 scores for all questions in the model, given the word tokens in a given question
        return self.model.get_scores(questiontokens)

    def return_score(self,q1,q2_idx):
        # return BM25 score for a particular question index in the model, given a new question
        return self.model.get_score(q1.tokens,q2_idx)
