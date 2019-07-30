
from nltk import word_tokenize

class Question:

    def __init__(self):
        self.id = False
        self.questiontext = False
        self.tokens = False
        self.emb = []
        # self.topics = False

    def import_qdict(self,qdict):
        self.id = qdict['id']
        self.questiontext = qdict['questiontext']
        self.tokens = qdict['tokens'] if 'tokens' in qdict.keys() else False
        #self.topics = self.import_topics(qdict['topics']) if 'topics' in qdict.keys() else {}

    def return_qdict(self,txt=True):
        qdict = {
            'id':self.id,
            'questiontext':self.questiontext,
            'tokens':self.tokens,
        }
        return qdict

    def tokenize(self):
        self.tokens = word_tokenize(self.questiontext,language='dutch')

    def encode(self,w2v):

        if len(self.emb) == 0:
            for w in self.tokens:
                try:
                    self.emb.append(w2v[w.lower()])
                except:
                    self.emb.append(300 * [0])
