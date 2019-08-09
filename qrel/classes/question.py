
class Question:
    """
    Class to manage all information related to a single question
    """

    def __init__(self):
        self.id = False
        self.questiontext = False
        self.tokens = False
        self.lemmas = False
        self.pos = False
        self.topics = False
        self.related = False
        self.emb = [] # not stored in qdict

    def import_qdict(self,qdict):
        # function to import json-formatted questions from file
        self.id = qdict['id']
        self.questiontext = qdict['questiontext']
        self.tokens = qdict['tokens'] if 'tokens' in qdict.keys() else False
        self.lemmas = qdict['lemmas'] if 'lemmas' in qdict.keys() else False
        self.pos = qdict['pos'] if 'pos' in qdict.keys() else False
        self.topics = qdict['topics'] if 'topics' in qdict.keys() else False
        self.related = qdict['related'] if 'related' in qdict.keys() else False

    def return_qdict(self,short=False):
        # function to return questions to json to write to output 
        if short:
            qdict = {
                'qid':self.id,
                'related':[{'questiontext':x[1],'qid':x[0]} for x in self.related]
            }
        else:
            qdict = {
                'id':self.id,
                'questiontext':self.questiontext,
                'tokens':self.tokens,
                'lemmas':self.lemmas,
                'pos':self.pos,
                'topics':self.topics,
                'related':self.related
            }
        return qdict

    def set_tokens(self,tokens):
        # function to connect tokens to question
        self.tokens = tokens

    def set_topics(self,topics):
        # function to connect extracted topics to question
        self.topics = topics

    def set_related(self,related):
        # function to connect related questions to question
        self.related = related

    def set_emb(self,emb):
        # function to connect word2vec embeddings of question tokens to question
        self.emb = emb

    def preprocess(self,nlp):
        """
        function to extract tokens, lemmas and part-of-speech tags from question, 
        which is needed for similarity prediction and topic extraction
        The Spacy nl_core_news_sm model is used for this
        """
        self.tokens, self.lemmas, self.pos = [],[],[]
        preprocessed = nlp(self.questiontext)
        for token in preprocessed:
            if not token.pos_ == 'PUNCT':
                self.tokens.append(token.text.lower())
                self.lemmas.append(token.lemma_)
                self.pos.append(token.pos_)
