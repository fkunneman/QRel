
import json
import sys

from topic_extraction import topic_extractor

commonness_dir = sys.argv[1]
entropy = sys.argv[2]
questions_frogged = sys.argv[3]
questions_meta = sys.argv[4]
questions_topics_out = sys.argv[5]

# read frogged questions
print('Reading frogged questions')
with open(questions_frogged,'r',encoding='utf-8') as file_in:
    qf = json.loads(file_in.read())
print('Done.',len(qf),'frogged questions')
tokens = []
for q in qf:
    tok = []
    for sen in q:
        tok.extend([x['text'] for x in sen])
    tokens.append(tok)

# read meta questions
print('Reading meta questions')
with open(questions_meta,'r',encoding='utf-8') as file_in:
    qm = json.loads(file_in.read())
print('Done.',len(qm),'meta questions')
for i,q in enumerate(qm):
    q['tokens'] = tokens[i]

# initialize topic extractor
print('Initializing topic extractor')
te = topic_extractor.TopicExtractor(commonness_dir,entropy)

# extract topics
print('Extracting topics')
questions_topics = te.extract_list(qf)
print('Done. Question_topics',len(questions_topics))

# write to output
print('Writing to output')
out = []
for i,q in enumerate(qm):
    q['topics'] = questions_topics[i]
    out.append(q)

with open(questions_topics_out,'w',encoding='utf-8') as file_out:
    json.dump(out,file_out)
