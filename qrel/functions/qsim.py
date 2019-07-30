
import os
import json

from qrel.classes import question, gv_bm25, trlm, softcosine, ensemble

class QSim:

    def __init__(self,questions,embeddings,d,tfidf,qprep,w2vpath):
        self.questions = questions
        self.embeddings = embeddings
        self.id2q = self.id2question_index()
        self.d = d
        self.tfidf = tfidf
        self.qprep = qprep
        self.w2vpath = w2vpath
        self.model = False
        self.gv_bm25 = False
        self.trlm = False
        self.softcosine = False
        self.ensemble = False

    def id2question_index(self):
        id2q = {}
        for i,q in enumerate(self.questions):
            id2q[q.id] = i
        return id2q

    def add_question(self,qdict):
        q = question.Question()
        q.import_qdict(qdict)
        self.questions.append(q)
        self.id2q[q.id] = len(self.questions)-1
        self.gv_bm25.init_model([q.tokens for q in self.questions])
        new_embeddings, q_emb = self.qprep.add_question(self.embeddings,q,self.w2vpath)
        self.embeddings = new_embeddings
        self.trlm.embeddings = new_embeddings
        self.softcosine.embeddings = new_embeddings
        q.set_emb(q_emb)

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
        self.trlm = trlm.TRLM(self.embeddings,self.d)
        if os.path.exists(modelpath):
            self.trlm.load_model(modelpath)
        else:
            print('File with TRLM model',trlmpath,'does not exist, training new model...')
            self.trlm.init_model([q.tokens for q in self.questions])
            print('Done. Saving model to',modelpath)
            self.trlm.save_model(modelpath)

    def init_softcosine(self):
        self.softcosine = softcosine.SoftCosine(self.embeddings,self.d,self.tfidf)

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
        return [bm25score,translation,softcosine]

    def train_model(self,traindata):
        trainvectors, labels = [], []
        for q1id in traindata:
            for q2id in traindata[q1id]:
                try:
                    q1 = self.questions[self.id2q[q1id]]
                except KeyError:
                    print('Question 1 with id',q1id,'not in data, adding...')
                    qdict = {'id':q1id,'questiontext':' '.join(traindata[q1id][q2id]['q1']),'tokens':traindata[q1id][q2id]['q1']}
                    self.add_question(qdict)
                    q1 = self.questions[self.id2q[q1id]]
                try:
                    q2 = self.questions[self.id2q[q2id]]
                except KeyError:
                    print('Question 2 with id',q2id,'not in data, adding...')
                    qdict = {'id':q2id,'questiontext':' '.join(traindata[q1id][q2id]['q2']),'tokens':traindata[q1id][q2id]['q1']}
                    self.add_question(qdict)
                    q2 = self.questions[self.id2q[q2id]]
                label = traindata[q1id][q2id]['label']

                print('Embeddings shape',self.embeddings.shape)
                print('Q1 TOKENS',' '.join(q1.tokens).encode('utf-8'),'Q1 EMB',q1.emb,'Q2 TOKENS',' '.join(q2.tokens).encode('utf-8'),'Q2 EMB',q2.emb)
                trainvectors.append(self.return_scores(q1,q2))
                labels.append(label)

        self.ensemble.train_regression(trainvectors=trainvectors, labels=labels, c='search', penalty='search', tol='search', gridsearch='brutal', jobs=10)

    def qsim(self,question1,question2):
        return self.ensemble.apply_model(self.return_scores(question1,question2))

    def retrieve_candidates(self,question,n):
        scores = self.gv_bm25.return_scores(question.tokens)
        scores_numbers = [[i,score] for i,score in enumerate(scores)]
        scores_numbers_ranked = sorted(scores_numbers,key = lambda k : k[1],reverse=True)
        return [self.questions[i] for i,score in scores_numbers_ranked[:n]]

    def rerank_candidates(self,q,candidates,approach='ensemble'):
        candidate_score = []
        if approach == 'bm25':
            for candidate in candidates:
                candidate_score.append([candidate,self.bm25.return_score(q,candidate)])
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


