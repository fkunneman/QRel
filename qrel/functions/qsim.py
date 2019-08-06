
import os
import json
import numpy as np

from qrel.classes import question, gv_bm25, trlm, softcosine, ensemble

class QSim:

    def __init__(self,questions,d,tfidf,w2v):
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

    def encode(self,tokens):
        emb = []
        for t in tokens:
            try:
                emb.append(self.w2v[t])
            except:
                emb.append(300 * [0])
        return emb

    def id2question(self):
        id2q = {}
        for i,q in enumerate(self.questions):
            id2q[q.id] = i
        return id2q

    def add_question(self,qdict):
        q = question.Question()
        q.import_qdict(qdict)
        q.set_emb(self.encode(q.tokens))
        self.questions.append(q)
        self.id2q[q.id] = len(self.questions)-1
        self.gv_bm25.init_model([q.tokens for q in self.questions])

    def init_bm25(self,modelpath):
        self.gv_bm25 = gv_bm25.GV_BM25()
        if os.path.exists(modelpath):
            self.gv_bm25.load_model(modelpath)
        else:
            print('File with BM25 model',modelpath,'does not exist, training new model...')
            self.gv_bm25.init_model([q.tokens for q in self.questions])
            print('Done. Saving model to',modelpath)
            self.gv_bm25.save_model(modelpath)

    def init_trlm(self,modelpath):
        self.trlm = trlm.TRLM(self.d)
        if os.path.exists(modelpath):
            self.trlm.load_model(modelpath)
        else:
            print('File with TRLM model',trlmpath,'does not exist, training new model...')
            self.trlm.init_model([q.tokens for q in self.questions])
            print('Done. Saving model to',modelpath)
            self.trlm.save_model(modelpath)

    def init_softcosine(self):
        self.softcosine = softcosine.SoftCosine(self.d,self.tfidf)

    def init_ensemble(self,ensemblepath,traindatapath):
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

    def return_scores(self,question1,question2):
        bm25score = self.gv_bm25.return_score(question1, self.id2q[question2.id])
        translation = self.trlm.apply_model(question1, question2)
        softcosine = self.softcosine.apply_model(question1, question2)
        if not np.isnan(softcosine):
            return [bm25score,translation,softcosine]
        else:
            return False

    def train_model(self,traindata):
        trainvectors, labels = [], []
        for q1id in traindata:
            for q2id in traindata[q1id]:
                try:
                    q1 = self.questions[self.id2q[q1id]]
                    q1.encode(self.w2v)
                except KeyError:
                    print('Question 1 with id',q1id,'not in data, adding...')
                    qdict = {'id':q1id,'questiontext':' '.join(traindata[q1id][q2id]['q1']),'tokens':[w.lower() for w in traindata[q1id][q2id]['q1']]}
                    self.add_question(qdict)
                    q1 = self.questions[self.id2q[q1id]]
                try:
                    q2 = self.questions[self.id2q[q2id]]
                    q2.encode(self.w2v)
                except KeyError:
                    print('Question 2 with id',q2id,'not in data, adding...')
                    qdict = {'id':q2id,'questiontext':' '.join(traindata[q1id][q2id]['q2']),'tokens':[w.lower() for w in traindata[q1id][q2id]['q1']]}
                    self.add_question(qdict)
                    q2 = self.questions[self.id2q[q2id]]
                label = traindata[q1id][q2id]['label']

                scores = self.return_scores(q1,q2)
                if scores:
                    trainvectors.append(scores)
                    labels.append(label)
                else:
                    continue

        self.ensemble.train_regression(trainvectors=trainvectors, labels=labels, c='search', penalty='search', tol='search', gridsearch='brutal', jobs=10)

    def qsim(self,question1,question2):
        return self.ensemble.apply_model(self.return_scores(question1,question2))

    def retrieve_candidates(self,questiontokens,n):
        scores = self.gv_bm25.return_scores(questiontokens)
        scores_numbers = [[i,score] for i,score in enumerate(scores)]
        scores_numbers_ranked = sorted(scores_numbers,key = lambda k : k[1],reverse=True)
        return [self.questions[i] for i,score in scores_numbers_ranked[:n]]

    def rerank_candidates(self,q,candidates,approach='ensemble'):
        candidate_score = []
        if approach == 'bm25':
            for candidate in candidates:
                candidate_score.append([candidate,self.bm25.return_score(q,candidate)])
        else:
            q.encode(self.w2v) 
            [c.encode(self.w2v) for c in candidates if len(c.emb) == 0]
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
