
import os
import json

import numpy
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, Word2Vec

from qrel.classes import question
from qrel.functions import qsim, qrel, topic_extractor

script_dir = os.path.dirname(__file__)
questionspath = script_dir + '/../../data/questions.json'
related_questionspath = script_dir + '/../../data/questions.related.json'
training_questionspath = script_dir + '/../../data/training_questions.json'
dictpath = script_dir + '/../../data/dict.model'
w2vpath = script_dir + '/../../data/word2vec.300_10.model'
tfidfpath = script_dir + '/../../data/tfidf.model'
trlmpath = script_dir + '/../../data/trlm.json'
ensemblepath = script_dir + '/../../data/ensemble.pkl'
commonness_path = script_dir + '/../../data/commonness_ngrams.txt'
entropy_path = script_dir + '/../../data/entropy_ngrams.txt'

class Relate:

    def __init__(self):

        self.questions = False
        self.qs = False
        self.topex = False
        self.qr = False

        self.load_questions()
        self.prepare_questions()
        self.init_qsim()
        self.init_topex()
        self.init_qrel()

    def __call__(self, qtext,qid,ncandidates=50):

        # prepare question object
        q = question.Question()
        q.questiontext = qtext
        q.id = qid
        q.preprocess()
        q.set_emb(self.qs.encode(q.tokens))
        q.set_topics(self.topex.extract(q))

        # retrieve related questions
        related = self.qr.relate_question(q,ncandidates=ncandidates)
        q.set_related(related)
        
        # update model
        self.update(q)
        
        return {'questiontext':qtext,'qid':qid,'related':related}

    def relate_many(self,questions):
        pass

    def update(self,q):
        self.questions.append(q)

    def test(self):

        questions_test = []
        for qobj in self.questions[-10:]:
            qobj.tokens = False
            qobj.lemmas = False
            qobj.pos = False
            qobj.topics = False
            questions_test.append(qobj)  
        self.questions = self.questions[:-10]
        self.init_qsim()
        self.init_qrel()

        print('Relating held-out questions to test')
        for q in questions_test:
            print('Question',q.questiontext.encode('utf-8'))
            print('Preprocessing question')
            q.preprocess()
            print('Extracting topics')
            topics = self.topex.extract(q)
            q.set_topics(topics)
            print('Topics','---'.join([t['topic'] for t in topics]).encode('utf-8'))
            print('Encoding question')
            emb = self.qs.encode(q.tokens)
            q.set_emb(emb)
            print('Retrieving similar questions')
            candidates = qs.retrieve_candidates(q.tokens,10)
            candidates_reranked_trlm = qs.rerank_candidates(q,candidates,approach='trlm')
            candidates_reranked_softcosine = qs.rerank_candidates(q,candidates,approach='softcosine')
            candidates_reranked_ensemble = qs.rerank_candidates(q,candidates)
            print('Candidates BM25:','---'.join([x.questiontext for x in candidates]).encode('utf-8'))
            print('Reranked TRLM:','---'.join([x[0].questiontext for x in candidates_reranked_trlm]).encode('utf-8'))
            print('Reranked SoftCosine:','---'.join([x[0].questiontext for x in candidates_reranked_softcosine]).encode('utf-8'))
            print('Reranked Ensemble','---'.join(['**'.join([x[0].questiontext,str(x[1]),str(x[2])]) for x in candidates_reranked_ensemble]).encode('utf-8'))
            print('Retrieving related questions')
            related = self.qr.relate_question(q)
            print('Related questions:')
            for r in related:
                print('***'.join([r[0].questiontext,str(r[1]),str(r[2])]).encode('utf-8'))

        print('Done. Restoring model')
        self.questions.extend(questions_test)
        self.init_qsim()
        self.init_qrel()


    def load_questions(self):

        # read in questions
        if os.path.exists(questionspath):
            print('Loading questions')
            with open(questionspath, 'r', encoding = 'utf-8') as file_in:
                questiondicts = json.loads(file_in.read())
            print('Formatting questions')
            self.questions = []
            for qd in questiondicts:
                qobj = question.Question()
                qobj.import_qdict(qd)
                self.questions.append(qobj)
      
        else:
            print('File with questions',questionspath,'does not exist, exiting program...')
            quit()

    def prepare_questions(self):

        # prepare questions
        if not self.questions[0].lemmas:
            print('Preprocessing questions, this may take a while...')
            counter = range(0,len(self.questions),1000)
            for i,q in enumerate(self.questions):
                if i in counter:
                    print('Question',i,'of',len(self.questions))
                q.preprocess()
            questions_preprocessed = [q.return_qdict() for q in questions]
            with open(questionspath,'w',encoding='utf-8') as file_out:
                json.dump(questions_preprocessed,file_out)

    def init_qsim(self):

        # initialize qsim
        d = Dictionary.load(dictpath)
        word2vec = Word2Vec.load(w2vpath)
        tfidf = TfidfModel.load(tfidfpath)
        self.qs = qsim.QSim(self.questions,d,tfidf,word2vec)
        print('Initializing BM25')
        self.qs.init_bm25()
        print('Initializing TRLM')
        self.qs.init_trlm(trlmpath)
        print('Initializing SoftCosine')
        self.qs.init_softcosine()
        print('Initializing Ensemble')
        self.qs.init_ensemble(ensemblepath,training_questionspath)

    def init_topex(self):

        # initialize topic extractor
        print('Initializing topic extractor')
        self.topex = topic_extractor.TopicExtractor(commonness_path,entropy_path)

    def init_qrel(self):

        # initialize question relator
        print('Initializing question relator')
        self.qr = qrel.QuestionRelator(self.qs)


