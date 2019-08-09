
import os
import json
import numpy as np

from qrel.classes import question, gv_bm25, trlm, softcosine, ensemble

class QSim:
    """
    Class to score the similarity between questions, using several models
    """

    def __init__(self,questions,d,tfidf,w2v):
        """
        Initialization of the model requeres (formatted) questions, a dictionary (d), 
        tfidf values and a trained word2vec model
        """
        self.questions = questions
        self.id2q = self.id2question()
        self.d = d
        self.tfidf = tfidf
        self.w2v = w2v
        self.model = False
        self.gv_bm25 = False
        self.trlm = False
        self.softcosine = False
        self.ensemble = False

    ############
    ### INIT ###
    ############

    def init_bm25(self):
        # Initialize BM25 model by storing question word tokens
        self.gv_bm25 = gv_bm25.GV_BM25()
        print('Training BM25model...')
        self.gv_bm25.init_model([q.tokens for q in self.questions])

    def init_trlm(self,modelpath):
        """
        Initialize translation-based language model, either by loading the model from a file 
        or training it based on the question tokens
        """
        self.trlm = trlm.TRLM(self.d)
        if os.path.exists(modelpath):
            self.trlm.load_model(modelpath)
        else:
            print('File with TRLM model',trlmpath,'does not exist, training new model...')
            self.trlm.init_model([q.tokens for q in self.questions])
            print('Done. Saving model to',modelpath)
            self.trlm.save_model(modelpath)

    def init_softcosine(self):
        # Initialize softcosine model by loading disctionary and tfidf
        self.softcosine = softcosine.SoftCosine(self.d,self.tfidf)

    def init_ensemble(self,ensemblepath,traindatapath):
        """
        Initialize ensemble model, either by loading the model from a file 
        or training it based on a file with training questions labeled for their similarity
        """
        self.ensemble = ensemble.Ensemble()
        if os.path.exists(ensemblepath): # load model
            print('Loading ensemble model')
            self.ensemble.load_model(ensemblepath)
        else: # train model
            print('Training ensemble model')
            with open(traindatapath,'r',encoding='utf-8') as file_in:
                traindata = json.loads(file_in.read())
            self.train_model(traindata['train'])
            self.ensemble.save_model(ensemblepath)


    ###############
    ### HELPERS ###
    ###############

    def train_model(self,traindata):
        """
        Function to prepare input to train an ensemble model based on training questions labeled for their similarity
        """
        trainvectors, labels = [], []
        for q1id in traindata:
            for q2id in traindata[q1id]:
                try:
                    q1 = self.questions[self.id2q[q1id]] # check if question1 is known
                    q1.encode(self.w2v)
                except KeyError:
                    print('Question 1 with id',q1id,'not in data, adding...')
                    # if question1 is not known, add to list
                    qdict = {'id':q1id,'questiontext':' '.join(traindata[q1id][q2id]['q1']),'tokens':[w.lower() for w in traindata[q1id][q2id]['q1']]}
                    self.add_question(qdict)
                    q1 = self.questions[self.id2q[q1id]]
                try:
                    q2 = self.questions[self.id2q[q2id]] # check if question2 is known
                    q2.encode(self.w2v)
                except KeyError:
                    print('Question 2 with id',q2id,'not in data, adding...')
                    # if question2 is not known, add to list
                    qdict = {'id':q2id,'questiontext':' '.join(traindata[q1id][q2id]['q2']),'tokens':[w.lower() for w in traindata[q1id][q2id]['q1']]}
                    self.add_question(qdict)
                    q2 = self.questions[self.id2q[q2id]]
                label = traindata[q1id][q2id]['label']

                # generate vector for pair of training questions by calculating BM25, Softcosine and TRLM
                scores = self.return_scores(q1,q2)
                if scores:
                    # store vector and label 
                    trainvectors.append(scores) 
                    labels.append(label)
                else:
                    continue

        # Train a logistic regression model based on trainvectors and labels, using brutal parameter tuning and 10 cores
        self.ensemble.train_regression(trainvectors=trainvectors, labels=labels, c='search', penalty='search', tol='search', gridsearch='brutal', jobs=10)

    def encode(self,tokens):
        """
        Function to encode question word tokens as a list of the word2vec dimensions per tokens
        """
        emb = []
        for t in tokens:
            try:
                emb.append(self.w2v[t]) # add the 300 dimensions of the word as stored in the word2vec model to the encoding
            except:
                emb.append(300 * [0]) # if the word is not known by the word2vec model, store as a list of 0's
        return emb

    def id2question(self):
        """
        Function to store the index of a question in the list by their question id
        needed for quickly finding question objects
        """
        id2q = {}
        for i,q in enumerate(self.questions):
            id2q[q.id] = i
        return id2q

    def add_question(self,qdict):
        """
        Function to add a question to the list of questions, needed for training a similarity model 
        Addition is done by importing the question dictionary and setting the word2vec encoding; the BM25 is updated as well
        """
        q = question.Question()
        q.import_qdict(qdict)
        q.set_emb(self.encode(q.tokens))
        self.questions.append(q)
        self.id2q[q.id] = len(self.questions)-1
        self.gv_bm25.init_model([q.tokens for q in self.questions])


    ##########################
    ### SIMILARITY SCORING ###
    ##########################

    def qsim(self,question1,question2):
        """
        Function to assess the similarity between two questions based on a trained ensemble model
        """
        try:
            return self.ensemble.apply_model(self.return_scores(question1,question2))
        except ValueError: # vectors not workable
            print('Could not calculate similarity between questions, returning 0-values')
            return [0.0,0]

    def return_scores(self,question1,question2):
        """
        Function to the similarity scores for two questions based on the three separate models
        (BM25, Softcosine and TRLM)
        """
        bm25score = self.gv_bm25.return_score(question1, self.id2q[question2.id])
        translation = self.trlm.apply_model(question1, question2)
        softcosine = self.softcosine.apply_model(question1, question2)
        if not np.isnan(softcosine):
            return [bm25score,translation,softcosine]
        else:
            return False

    def retrieve_candidates(self,questiontokens,n):
        """
        Function to retrieve candidate similar questions to a given set of question word tokens based on bm25
        In a subsequent step these candidates could be reranked by other (more time-consuming) metrics
        'n' gives the number of candidates to return
        """
        scores = self.gv_bm25.return_scores(questiontokens)
        scores_numbers = [[i,score] for i,score in enumerate(scores)]
        scores_numbers_ranked = sorted(scores_numbers,key = lambda k : k[1],reverse=True)
        return [self.questions[i] for i,score in scores_numbers_ranked[:n]]

    def rerank_candidates(self,q,candidates,approach='ensemble'):
        """
        Function to rerank a set of candidate similar questions to given questions, based on a chosen model 
        """
        candidate_score = []
        if approach == 'bm25':
            for candidate in candidates:
                candidate_score.append([candidate,self.bm25.return_score(q,candidate)])
        else:
            if len(q.emb) == 0:
                q.set_emb(self.encode(q.tokens))
            for c in candidates:
                if len(c.emb) == 0:
                    c.set_emb(self.encode(c.tokens))
            if approach == 'trlm':
                for candidate in candidates:
                    candidate_score.append([candidate,self.trlm.apply_model(q,candidate)])
            elif approach == 'softcosine':
                for candidate in candidates:
                    candidate_score.append([candidate,self.softcosine.apply_model(q,candidate)])
            elif approach == 'ensemble':
                for candidate in candidates:
                    candidate_score.append([candidate] + self.qsim(q,candidate))
        return sorted(candidate_score,key = lambda k : k[1],reverse = True)
