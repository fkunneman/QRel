__author__='floriankunneman'

import random
import sys
import json
import multiprocessing

import question_relator
from main import GoeieVraag

outfile = sys.argv[1]
outfile_shallow = sys.argv[2]
topic_percentage = float(sys.argv[3]) # topics to select
ncandidates = int(sys.argv[4]) # total questions to query

def make_chunks(lines,nc=12):
    i=0
    chunks=[]
    size = int(len(lines)/nc)
    for j in range(nc-1):
        chunks.append(lines[i:(i+size)])
        i += size
    chunks.append(lines[i:])
    return chunks

def relate(questions,c,i):
    for question in questions:
        output_question = [question['id'],question['questiontext']]
    # try:
        candidates_prominent_topics_popularity = qr.relate_question(question,topic_percentage,ncandidates,model='topic_all')
    #except:
    #    print('Error in question_relator, continuing to next question')
    #    continue
        all_topics = ['PROMINENT TOPICS']
        shallow = {'qid':question['id']}
        shallow_related = []
        for x in candidates_prominent_topics_popularity:
            all_topics.append({'id':x[0],'text':sim_model.seeds_text[x[0]],'topic':x[-2],'score':x[-1],'pop':x[3]})
            shallow_related.append({'qid':x[0],'questiontext':sim_model.seeds_text[x[0]]})
        shallow['related'] = shallow_related
        c.put([[output_question,all_topics],shallow])
    print('Chunk',i,'Done.')

# init
print('Initializing Similarity classifier')
sim_model = GoeieVraag(evaluation=False, w2v_dim=300)

print('Initializing Question relator')
qr = question_relator.QuestionRelator(sim_model)
qr.load_corpus()

# relate questions
print('Relating questions')
output = []
all_questions = len(sim_model.seeds)
chunks = make_chunks(sim_model.seeds)
q = multiprocessing.Queue()
for i in range(len(chunks)):
    p = multiprocessing.Process(target=relate,args=[chunks[i],q,i])
    p.start()

while True:
    l = q.get()
    output.append(l)
    print(len(output),'of',all_questions)
    if len(output) == all_questions:
        break

print('Done.')
full = [x[0] for x in output]
shallow = [x[1] for x in output]
with open(outfile,'w',encoding='utf=8') as out:
    json.dump(full,out)

with open(outfile_shallow,'w',encoding='utf-8') as out:
    json.dump(shallow,out)
