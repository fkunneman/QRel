#title           :relate.py
#description     :Container of all modules that communicate with the functions and classes in the QREL repo to update and maintain a database of related questions
#author          :Florian Kunneman, Thiago Castro Ferreira
#date            :20190809
#version         :0.1
#usage           :python relate.py test; python relate.py test_many; python relate.py [new_questions.json]
#notes           : 
#python_version  :3.5.2  
#==============================================================================

import os
import sys
import json
import warnings

import numpy
import spacy
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

warnings.filterwarnings("ignore")

class Relate:
    """
    Container of all modules that communicate with the functions and classes in the QREL 
        repo to update and maintain a database of related questions
    """

    def __init__(self):
        """
        All relevant models are initialized based on the file paths specified above of this class
        """
        self.questions = []
        self.candidates = []
        self.qs = False
        self.topex = False
        self.qr = False

        self.nlp = spacy.load('nl_core_news_sm')
        self.nlp.disable_pipes('parser','ner')
        self.load_questions()
        self.init_topex()
        self.prepare_questions()
        self.init_qsim()
        self.init_qrel()

    ############
    ### INIT ###
    ############

    def load_questions(self,qpath=questionspath):
        # read in questions
        if os.path.exists(qpath): # make sure that file in path exists
            print('Loading questions')
            with open(qpath, 'r', encoding = 'utf-8') as file_in:
                questiondicts = json.loads(file_in.read())
            print('Formatting questions')
            # format as question objects
            for qd in questiondicts:
                qobj = question.Question()
                qobj.import_qdict(qd)
                self.questions.append(qobj)
      
        else:
            print('File with questions',qpath,'does not exist, exiting program...')
            quit()

    def init_topex(self):
        # initialize topic extractor 
        print('Initializing topic extractor')
        self.topex = topic_extractor.TopicExtractor(commonness_path,entropy_path)

    def prepare_questions(self,index=0):
        # prepare questions - make sure they are preprocessed and their topics are extracted
        if not self.questions[index].lemmas: # rudimental check if preprocessing has already been done
            print('Preprocessing questions, this may take a while...')
            counter = range(0,len(self.questions)-index,100)
            for i,q in enumerate(self.questions[index:]):
                if i in counter:
                    print('Question',i,'of',len(self.questions)-index,'(counting per 100)')
                q.preprocess(self.nlp)
            self.save()
        if not self.questions[index].topics: # rudimental step to check if topics have been extracted
            print('Extracting topics from questions, this may take a while...')
            counter = range(0,len(self.questions)-index,100)
            for i,q in enumerate(self.questions[index:]):
                if i in counter:
                    print('Question',i,'of',len(self.questions)-index,'(counting per 100)')
                q.set_topics(self.topex.extract(q))
            self.save()
                
    def init_qsim(self,bm25only=False):
        # initialize qsim

        # fast init
        if bm25only:
            self.qs.questions = self.questions
            self.qs.init_bm25()
            self.qs.id2q = self.qs.id2question()
        # complete init
        else:
            # load models needed for initialization of qsim
            d = Dictionary.load(dictpath)
            word2vec = Word2Vec.load(w2vpath)
            tfidf = TfidfModel.load(tfidfpath)
            self.qs = qsim.QSim(self.questions,d,tfidf,word2vec)
            # initialize separate components of qsim
            print('Initializing BM25')
            self.qs.init_bm25()
            print('Initializing TRLM')
            self.qs.init_trlm(trlmpath)
            print('Initializing SoftCosine')
            self.qs.init_softcosine()
            print('Initializing Ensemble')
            self.qs.init_ensemble(ensemblepath,training_questionspath)

    def init_qrel(self):
        # initialize question relator
        print('Initializing question relator')
        self.qr = qrel.QuestionRelator(self.qs)


    ##############
    ### RELATE ###
    ##############

    def __call__(self,qtext,qid,ncandidates=50):
        """
        Generic model call by which related questions to a given question are retrieved from the dataset,
            and the dataset is updated
        """

        # prepare question object
        q = question.Question()
        q.questiontext = qtext
        q.id = qid
        q.preprocess(self.nlp)
        q.set_emb(self.qs.encode(q.tokens))
        q.set_topics(self.topex.extract(q))

        # retrieve related questions
        related, candidates = self.qr.relate_question(q,ncandidates=ncandidates)
        q.set_related(related)
        
        # update model
        self.update(q,candidates)
        
        return {'questiontext':qtext,'qid':qid,'related':related}

    def most_similar(self,qtext,model='ensemble'):
        """
        Generic model call by which the most similar questions to a given question are retrieved from the dataset
        """

        # prepare question object
        q = question.Question()
        q.questiontext = qtext
        q.preprocess(self.nlp)
        q.set_emb(self.qs.encode(q.tokens))

        # retrieve most similar questions
        candidates = self.qs.retrieve_candidates(q.tokens,15)
        similar = self.qs.rerank_candidates(q,candidates,approach=model)
        
        if model == 'ensemble':
            return {'questiontext':qtext, 'similar':[[x[0].id,x[0].questiontext,x[1],int(x[2])] for x in similar[:5]]}
        else:
            return {'questiontext':qtext, 'similar':[[x[0].id,x[0].questiontext,x[1],0] for x in similar[:5]]}

    def relate_many(self,qpath):
        """
        Function to update the dataset with a larger set of questions from a file (specified in qpath)
        """
        index = len(self.questions) # the current number of questions is stored to prevent redundant computations 
        self.load_questions(qpath) # add questions to current questions
        self.prepare_questions(index) # if questions do not contain preprocessed and/or topic information, apply these procedures
        self.init_qsim(bm25only=True) # reinitialize question similarity model with added questions
        self.init_qrel() # reinitialize question relatedness model with added questions
        redo = [] # list to store questions in original dataset that might need their related questions updated
        print('Relating new questions, this may take a while...')
        counter = range(0,len(self.questions)-index,100)
        for i,q in enumerate(self.questions[index:]):
            if i in counter:
                print('Question',i,'of',len(self.questions)-index,'(counting per 100)')
            related, candidates = self.qr.relate_question(q) # apply question relatedness
            q.set_related(related)
            redo.extend(candidates) # store all candidates related to the current question, their question relatedness will be updated later
        # update related questions for original questions
        self.candidates = [c for c in list(set(redo)) if self.qs.id2q[c] < index]
        self.update_candidates()
        
    def update(self,q,candidates):
        # function to update models with added question
        self.questions.append(q)
        self.candidates.extend(candidates)
        self.init_qsim(bm25only=True) # reinitialize question similarity model with added question
        self.init_qrel()

    def save(self):
        # function to write current questions to file
        print('Overwriting files with current dataset')
        extended_questions_json = [q.return_qdict() for q in self.questions]
        with open(questionspath,'w',encoding='utf-8') as file_out: # store updated questions to file
            json.dump(extended_questions_json,file_out)
        try:
            related_questions_json = [q.return_qdict(short=True) for q in self.questions]
            with open(related_questionspath,'w',encoding='utf-8') as file_out:
                json.dump(related_questions_json,file_out)
        except:
            pass
                
    def update_candidates(self):
        print('Updating related questions for candidates, this may take a while...')
        # update related questions for candidates
        counter = range(0,len(self.candidates),100)
        for i,c in enumerate(self.candidates):
            if i in counter:
                print('Question',i,'of',len(self.candidates),'(counting per 100)')
            cq = self.questions[self.qs.id2q[c]]
            related, candidates = self.qr.relate_question(cq)
            cq.set_related(related)
        print('Done. Writing data')
        self.save()
        self.candidates = []
        

    ############
    ### TEST ###
    ############

    def test_relate(self):
        """
        Function to test question similarity and relatedness procedure by running them on 10 held-out questions
        """
        questions_test = []
        for qobj in self.questions[-10:]: # select held-out questions and remove preprocessing and topic information
            qobj.tokens = False
            qobj.lemmas = False
            qobj.pos = False
            qobj.topics = False
            questions_test.append(qobj)  
        self.questions = self.questions[:-10] # strip held-out questions from original questions
        # reinitialize models
        self.init_qsim()
        self.init_qrel()

        print('Relating held-out questions to test')
        for q in questions_test: # for each held-out question
            print('Question',q.questiontext.encode('utf-8')) # print questiontext to screen
            # preprocess and extract topics
            print('Preprocessing question')
            q.preprocess(self.nlp) 
            print('Extracting topics')
            topics = self.topex.extract(q)
            q.set_topics(topics)
            print('Topics','---'.join([t['topic'] for t in topics]).encode('utf-8')) # print topics to screen
            print('Encoding question')
            emb = self.qs.encode(q.tokens)
            q.set_emb(emb)
            # test all question similarity models and print output to screen
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
            # test question relatedness and print to screen
            print('Related questions:')
            for r in related:
                print('***'.join([r[0].questiontext,str(r[1]),str(r[2])]).encode('utf-8'))

        # restore models with held-out questions added to original set again
        print('Done. Restoring model') 
        self.questions.extend(questions_test)
        self.init_qsim()
        self.init_qrel()

    def test_relate_many(self):
        """
        Function to test question similarity and relatedness procedure for a new file with many questions
        """
        many_questions_test = []
        for qobj in self.questions[-100:]: # select held-out questions and remove preprocessing and topic information
            qobj.tokens = False
            qobj.lemmas = False
            qobj.pos = False
            qobj.topics = False
            many_questions_test.append(qobj)  
        self.questions = self.questions[:-100] # strip held-out questions from original questions
        # reinitialize models
        self.init_qsim()
        self.init_qrel()

        # write held-out questions to new file
        print('Writing held-out questions to dummy file')
        many_questions_test_formatted = [q.return_qdict() for q in many_questions_test]
        qpath = questionspath[:-4] + 'dummy.json'
        with open(qpath,'w',encoding='utf-8') as file_out:
            json.dump(many_questions_test_formatted,file_out)
                        
        # run relate_many function on new file
        print('Relating held-out questions from dummy file')
        self.relate_many(qpath)


    ############
    ### MAIN ###
    ############

if __name__ == '__main__':
    """ 
    Run one of the test functions, or relate many questions by passing the name of a file with questions as argument
    usage: 
        python qrel/modules/relate.py test
        python qrel/modules/relate.py test_many
        python qrel/modules/relate.py [new_questions.json]
    """
    arg = sys.argv[1]
    model = Relate()
    if arg == 'test':
        model.test_relate()
    elif arg == 'test_many':
        model.test_relate_many()
    else:
        model.relate_many(arg)
