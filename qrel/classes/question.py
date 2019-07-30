
from nltk import word_tokenize

class Question:

    def __init__(self):
        self.id = False
        self.questiontext = False
        self.tokens = False
        self.emb = False
        # self.topics = False

    def import_qdict(self,qdict):
        self.id = qdict['id']
        self.questiontext = qdict['questiontext']
        self.tokens = qdict['tokens'] if 'tokens' in qdict.keys() else False
        self.emb = qdict['emb'] if 'emb' in qdict.keys() else False
        #self.topics = self.import_topics(qdict['topics']) if 'topics' in qdict.keys() else {}

    def return_qdict(self,txt=True):
        qdict = {
            'id':self.id,
            'questiontext':self.questiontext,
            'tokens':self.tokens,
            'emb':self.emb
        }
        return qdict

    def tokenize(self):
        self.tokens = word_tokenize(self.questiontext,language='dutch')

    def set_emb(self,emb):
        self.emb = emb

    def extract_topics(self):
        pass
