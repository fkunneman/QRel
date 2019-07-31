
import spacy

class Question:

    def __init__(self):
        self.id = False
        self.questiontext = False
        self.tokens = False
        self.lemmas = False
        self.pos = False
        self.topics = False
        self.emb = [] # not stored in qdict

    def import_qdict(self,qdict):
        self.id = qdict['id']
        self.questiontext = qdict['questiontext']
        self.tokens = qdict['tokens'] if 'tokens' in qdict.keys() else False
        self.lemmas = qdict['lemmas'] if 'lemmas' in qdict.keys() else False
        self.pos = qdict['pos'] if 'pos' in qdict.keys() else False
        self.topics = qdict['topics'] if 'topics' in qdict.keys() else False

    def return_qdict(self,txt=True):
        qdict = {
            'id':self.id,
            'questiontext':self.questiontext,
            'tokens':self.tokens,
            'lemmas':self.lemmas,
            'pos':self.pos,
            'topics':self.topics
        }
        return qdict

    def set_topics(self,topics):
        self.topics = topics

    def preprocess(self):
        nlp = spacy.load('nl_core_news_sm')
        self.tokens, self.lemmas, self.pos = [],[],[]
        preprocessed = nlp(self.questiontext)
        for token in preprocessed:
            if not token.pos_ == 'PUNCT':
                self.tokens.append(token.text.lower())
                self.lemmas.append(token.lemma_)
                self.pos.append(token.pos_)
