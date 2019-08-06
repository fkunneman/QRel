
import os
import json

import numpy
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, Word2Vec

from qrel.classes import question
from qrel.functions import qsim, qrel, topic_extractor

script_dir = os.path.dirname(__file__)
questionspath = script_dir + '/../../data/questions.json'
training_questionspath = script_dir + '/../../data/training_questions.json'
dictpath = script_dir + '/../../data/dict.model'
w2vpath = script_dir + '/../../data/word2vec.300_10.model'
tfidfpath = script_dir + '/../../data/tfidf.model'
trlmpath = script_dir + '/../../data/trlm.json'
ensemblepath = script_dir + '/../../data/ensemble.pkl'
commonness_path = script_dir + '/../../data/commonness_ngrams.txt'
entropy_path = script_dir + '/../../data/entropy_ngrams.txt'

# read in questions
if os.path.exists(questionspath):
    print('Loading questions')
    with open(questionspath, 'r', encoding = 'utf-8') as file_in:
        questiondicts = json.loads(file_in.read())
    print('Formatting questions')
    questions = []
    for qd in questiondicts[:-10]:
        qobj = question.Question()
        qobj.import_qdict(qd)
        questions.append(qobj)
    questions_test = []
    for qd in questiondicts[-10:]:
        qobj = question.Question()
        qobj.import_qdict(qd)
        qobj.tokens = False
        qobj.lemmas = False
        qobj.pos = False
        qobj.topics = False
        questions_test.append(qobj)        

else:
    print('File with questions',questionspath,'does not exist, exiting program...')
    quit()

# prepare questions
if not questions[0].lemmas:
    print('Preprocessing questions')
    for q in questions:
        q.preprocess()
    questions_preprocessed = [q.return_qdict() for q in questions]
    with open(questionspath,'w',encoding='utf-8') as file_out:
        json.dump(questions_preprocessed,file_out)

# initialize qsim
d = Dictionary.load(dictpath)
word2vec = Word2Vec.load(w2vpath)
tfidf = TfidfModel.load(tfidfpath)
qs = qsim.QSim(questions,d,tfidf,word2vec)
print('Initializing BM25')
qs.init_bm25()
print('Initializing TRLM')
qs.init_trlm(trlmpath)
print('Initializing SoftCosine')
qs.init_softcosine()
print('Initializing Ensemble')
qs.init_ensemble(ensemblepath,training_questionspath)

# initialize topic extractor
print('Initializing topic extractor')
topex = topic_extractor.TopicExtractor(commonness_path,entropy_path)

# initialize question relator
print('Initializing question relator')
qr = qrel.QuestionRelator(qs)

# prepare test questions
print('Relating held-out questions')
for q in questions_test:
    print('Question',q.questiontext.encode('utf-8'))
    print('Preprocessing question')
    q.preprocess()
    print('Extracting topics')
    topics = topex.extract(q)
    q.set_topics(topics)
    print('Topics','---'.join([t['topic'] for t in topics]).encode('utf-8'))
    print('Encoding question')
    emb = qs.encode(q.tokens)
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
    related = qr.relate_question(q)
    print('Related questions:')
    for r in related:
        print('***'.join([r[0].questiontext,str(r[1]),str(r[2])]).encode('utf-8'))
