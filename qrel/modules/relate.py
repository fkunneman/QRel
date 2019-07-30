
import os
import json

import numpy
from scipy import sparse
from gensim.corpora import Dictionary
from gensim.models import TfidfModel

from qrel.classes import question
from qrel.functions import qprep, qsim

script_dir = os.path.dirname(__file__)
questionspath = script_dir + '/../../data/questions.json'
encoded_questionspath = script_dir + '/../../data/encoded_questions.npz'
training_questionspath = script_dir + '/../../data/training_questions.json'
dictpath = script_dir + '/../../data/dict.model'
w2vpath = script_dir + '/../../data/word2vec.300_10.model'
tfidfpath = script_dir + '/../../data/tfidf.model'
bm25path = script_dir + '/../../data/bm25.pkl'
trlmpath = script_dir + '/../../data/trlm.json'
ensemblepath = script_dir + '/../../data/ensemble.pkl'

# read in questions
if os.path.exists(questionspath):
    print('Loading questions')
    with open(questionspath, 'r', encoding = 'utf-8') as file_in:
        questiondicts = json.loads(file_in.read())
    print('Formatting questions')
    questions = []
    for qd in questiondicts:
        qobj = question.Question()
        qobj.import_qdict(qd)
        questions.append(qobj)

else:
    print('File with questions',questionspath,'does not exist, exiting program...')
    quit()

# prepare questions
qp = qprep.QPrep(questions)
altered = False
if not questions[0].tokens:
    qp.tokenize_questions()
    questions_formatted = [q.return_qdict() for q in qp.questions]
    with open(questionspath,'w',encoding='utf-8') as file_out:
        json.dump(questions_formatted,file_out)
if os.path.exists(encoded_questionspath):
    print('Loading question embeddings')
    loader = numpy.load(encoded_questionspath)
    encoded_questions = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])
else:
    print('Encoding questions')
    encoded_questions, encoded_questions_indices = qp.encode_questions(w2vpath)
    print('Done, saving to file.')
    for i,q in enumerate(questions):
        q.set_emb(encoded_questions_indices[i])
    questions_formatted = [q.return_qdict() for q in qp.questions]
    with open(questionspath,'w',encoding='utf-8') as file_out:
        json.dump(questions_formatted,file_out)
    numpy.savez(encoded_questionspath, data=encoded_questions.data, indices=encoded_questions.indices, indptr=encoded_questions.indptr, shape=encoded_questions.shape)

# initialize qsim
d = Dictionary.load(dictpath)
tfidf = TfidfModel.load(tfidfpath)
qs = qsim.QSim(questions,encoded_questions,d,tfidf,qp,w2vpath)
print('Initializing BM25')
qs.init_bm25(bm25path)
print('Initializing TRLM')
qs.init_trlm(trlmpath)
print('Initializing SoftCosine')
qs.init_softcosine()
print('Initializing Ensemble')
qs.init_ensemble(ensemblepath,training_questionspath)

candidates = qs.retrieve_candidates(questions[0],10)
candidates_reranked_trlm = qs.rerank_candidates(questions[0],candidates,approach='trlm')
candidates_reranked_softcosine = qs.rerank_candidates(questions[0],candidates,approach='softcosine')
candidates_reranked_ensemble = qs.rerank_candidates(questions[0],candidates)
print('Seed question:',questions[0].questiontext.encode('utf-8'))
print('Candidates BM25:','---'.join([x.questiontext for x in candidates]).encode('utf-8'))
print('Reranked TRLM:','---'.join([x[0].questiontext for x in candidates_reranked_trlm]).encode('utf-8'))
print('Reranked SoftCosine:','---'.join([x[0].questiontext for x in candidates_reranked_softcosine]).encode('utf-8'))
print('Reranked Ensemble','---'.join(['**'.join([x[0].questiontext,str(x[1]),str(x[2])]) for x in qs.candidates_reranked_ensemble]).encode('utf-8'))



    