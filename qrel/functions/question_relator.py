
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

class QuestionRelator:

    def __init__(self,sim_model):
        self.sim_model = sim_model
            
    def load_corpus(self):
        self.qid_starcount = {}
        self.qid_text = {}
        self.qid_topics = {}
        self.topic_emb = {}
        for question in self.sim_model.seeds:
            self.qid_starcount[question['id']] = question['starcount']
            # self.qid_text[question['id']] = question['text']
            self.qid_text[question['id']] = question['questiontext']
            self.qid_topics[question['id']] = question['topics']
            for topic in question['topics']:
                self.topic_emb[topic['topic']] = False            

    def softcosine_topics(self, t1, t1emb, q2, q2emb):

        def dot(q1tfidf, q1emb, q2tfidf, q2emb):
            cos = 0.0
            for i, w1 in enumerate(q1tfidf):
                for j, w2 in enumerate(q2tfidf):
                    if w1[0] == w2[0]:
                        cos += (w1[1] * w2[1])
                    else:
                        m_ij = max(0, cosine_similarity([q1emb[i]], [q2emb[j]])[0][0])**2
                        cos += (w1[1] * m_ij * w2[1])
            return cos

        t1tfidf = self.sim_model.tfidf[self.sim_model.dict.doc2bow(t1)]
        t1t1 = np.sqrt(dot(t1tfidf, t1emb, t1tfidf, t1emb))

        softcosines = []
        for i,t2 in enumerate(q2):
            t2tfidf = self.sim_model.tfidf[self.sim_model.dict.doc2bow([t2])]            
            t2t2 = np.sqrt(dot(t2tfidf, [q2emb[i]], t2tfidf, [q2emb[i]]))
            softcosines.append(dot(t1tfidf, t1emb, t2tfidf, [q2emb[i]]) / (t1t1 * t2t2))
            
        softcosine = np.mean([x for x in softcosines if not np.isnan(x)])
        return softcosine

    def softcosine_topics2(self, t1, ts2):

        def dot(q1tfidf, q1emb, q2tfidf, q2emb):
            cos = 0.0
            for i, w1 in enumerate(q1tfidf):
                for j, w2 in enumerate(q2tfidf):
                    if w1[0] == w2[0]:
                        cos += (w1[1] * w2[1])
                    else:
                        m_ij = max(0, cosine_similarity([q1emb[i]], [q2emb[j]])[0][0])**2
                        cos += (w1[1] * m_ij * w2[1])
            return cos

        print('T1',' '.join(t1).encode('utf-8'))
        print('T2',' '.join(t2).encode('utf-8'))
        t1tfidf = self.sim_model.tfidf[self.sim_model.dict.doc2bow(t1)]
        t1t1 = np.sqrt(dot(t1tfidf, t1emb, t1tfidf, t1emb))
        t2tfidf = self.sim_model.tfidf[self.sim_model.dict.doc2bow(t2)]
        t2t2 = np.sqrt(dot(t2tfidf, t2emb, t2tfidf, t2emb))
        softcosine = dot(t1tfidf, t1emb, t2tfidf, t2emb) / (t1t1 * t2t2)

        return softcosine

    def return_prominent_topics(self,topics,percentage=0.80):
        total = sum([x['topic_score'] for x in topics])
        if total == 0:
            return []
        acc = 0
        selection = []
        for topic in topics:
            selection.append(topic)
            acc += topic['topic_score']
            if (acc/total > percentage):
                break
        return selection

    def deduplicate_bm25_output(self,q_index,bm25_output):
        qids = []
        deduplicated = []
        for retrieved in bm25_output:
            rid = self.sim_model.idx2id[retrieved[0]]
            if rid != q_index:
                if rid not in qids:
                    qids.append(rid)
                    deduplicated.append(retrieved+[rid])
        return deduplicated
                    
    def select_candidates_bm25(self,question,topics,ntargets):
        # print('BM25 for',' '.join(question['tokens']).encode('utf-8'))
        scores_all = self.sim_model.bm25.get_scores(question['tokens'])
        numbers_scores = [[i,score] for i,score in enumerate(scores_all)]
        if len(topics) == 1:
            ntargets_by_chunk = ntargets
        else:
            ntargets_by_chunk = int(ntargets / (len(topics)+1))
        # ntargets_by_chunk = int(ntargets / (len(topics)+1))
        numbers_scores_ranked = sorted(numbers_scores,key = lambda k : k[1],reverse=True)[:ntargets_by_chunk]
        scores = []
        scores.extend(numbers_scores_ranked)
        if len(topics) > 1:
            for i,topic in enumerate(topics):
                # print('BM25 for',' '.join(list(set(question['tokens']) - set(topic['topic_text'].split()))).encode('utf-8'))
                # other_topics_tokens = sum([x['topic_text'].split() for j,x in enumerate(topics) if j != i],[])
                # scores_all = self.sim_model.bm25.get_scores(list(set(question['tokens']) - set(other_topics_tokens)))
                scores_all = self.sim_model.bm25.get_scores(list(set(question['tokens']) - set(topic['topic_text'].split())))
                numbers_scores = [[i,score] for i,score in enumerate(scores_all)]
                numbers_scores_ranked = sorted(numbers_scores,key = lambda k : k[1],reverse=True)[:ntargets_by_chunk]
                # print('Found','\n'.join([' '.join(self.sim_model.qid_tokens[self.sim_model.idx2id[i[0]]]) for i in numbers_scores_ranked]).encode('utf-8'))
                scores.extend(numbers_scores_ranked)

                # print('BM25 for',' '.join(topic['topic_text']).encode('utf-8'))
                # other_topics_tokens = sum([x['topic_text'].split() for j,x in enumerate(topics) if j != i],[])
                # scores_all = self.sim_model.bm25.get_scores(list(set(question['tokens']) - set(other_topics_tokens)))
            # scores_all = self.sim_model.bm25.get_scores(topic['topic_text'].split())
            # numbers_scores = [[i,score] for i,score in enumerate(scores_all)]
            # numbers_scores_ranked = sorted(numbers_scores,key = lambda k : k[1],reverse=True)[:ntargets_by_chunk]
            #     # print('Found','\n'.join([' '.join(self.sim_model.qid_tokens[self.sim_model.idx2id[i[0]]]) for i in numbers_scores_ranked]).encode('utf-8'))
            # scores.extend(numbers_scores_ranked)

        scores_ranked = sorted(scores,key = lambda k : k[1],reverse=True)
        scores_deduplicated = self.deduplicate_bm25_output(question['id'],scores_ranked)
        return scores_deduplicated

    def rank_classify_similarity(self,question,targets):
        targets_score_clf = []
        for i,target in enumerate(targets):
            q2id = target[-1]
            try:
                q2 = self.sim_model.qid_tokens[q2id]
                q2emb = self.sim_model.encode(q2)
                clfscore, pred_label = self.sim_model.ensembling(question['tokens'],question['emb'], q2id, q2, q2emb)
            except:
                print('QID not in seeds,continueing')
                continue
            try:
                targets_score_clf.append([q2id,q2,q2emb,self.qid_starcount[q2id],clfscore,int(pred_label)])
            except:
                targets_score_clf.append([q2id,q2,q2emb,0,clfscore,int(pred_label)])
                
        targets_score_clf_ranked = sorted(targets_score_clf,key = lambda k : (k[4],k[3]),reverse=True)
        return targets_score_clf_ranked
    
    def rank_questions_topics(self,question,topics,targets):
        ranked_by_topic = []
        for topic in topics:
            tokens = topic['topic'].split()
            if not self.topic_emb[topic['topic']]:
                self.topic_emb[topic['topic']] = self.sim_model.encode(tokens)
            emb = self.topic_emb[topic['topic']]
            topic_sims = []
            for target in targets:
                # sim = self.softcosine_topics(tokens, emb, target[1], target[2])
                sim = self.sim_model.softcos(tokens, emb, target[1], target[2])
                topic_sims.append(target + [topic['topic'],sim])
            topic_sims_ranked = sorted(topic_sims,key = lambda k : k[-1],reverse=True)
            ranked_by_topic.append(topic_sims_ranked)
        return ranked_by_topic

    def rank_questions_topics2(self,question,topics,targets):
        ranked_by_topic = []
        for topic in topics:
            print('QTOPIC',topic['topic_text'].encode('utf-8'))
            tokens = topic['topic_text'].split()
            if not self.topic_emb[topic['topic_text']]:
                self.topic_emb[topic['topic_text']] = self.sim_model.encode(tokens)
            emb = self.topic_emb[topic['topic_text']]
            topic_sims = []
            for target in targets:
                print('Target',' '.join(target[1]).encode('utf-8'))
                sims = []
                for ttopic in self.qid_topics[target[0]]:
                    if ttopic['topic_score'] > 0.15:
                        print('Target Topic',ttopic['topic_text'].encode('utf-8'))
                        if not self.topic_emb[ttopic['topic_text']]:
                            self.topic_emb[ttopic['topic_text']] = self.sim_model.encode(target[1])
                        sim = self.sim_model.softcos(tokens, emb, ttopic['topic_text'].split(), self.topic_emb[ttopic['topic_text']])
                        print('SIM',sim)
                        sims.append(sim)
                if len(sims) > 0:
                    print('MAX SIM',max(sims))
                    topic_sims.append(target + [topic['topic_text'],max(sims)])
            topic_sims_ranked = sorted(topic_sims,key = lambda k : k[-1],reverse=True)
            ranked_by_topic.append(topic_sims_ranked)
        return ranked_by_topic

    def diversify(self,targets,diversity_threshold):
        diversified = []
        for target in targets:
            if len(diversified) == 0:
                diversified.append(target)
            else:
                diverse = True
                for target2 in diversified:
                    softcosine = self.sim_model.softcos(target2[1],target2[2],target[1],target[2])
                    if softcosine > diversity_threshold:
                        diverse = False
                if diverse:
                    diversified.append(target)
        return diversified
    
    def relate_question(self,question,topic_percentage,ncandidates,model='pack'):

        question['emb'] = self.sim_model.encode(question['tokens'])

        # select prominent topics
        prominent_topics = self.return_prominent_topics(question['topics'],topic_percentage)

        # retrieve candidate_questions
        candidates = self.select_candidates_bm25(question,prominent_topics,ncandidates)

        # score and rank questions by similarity
        candidates_ranked_sim = self.rank_classify_similarity(question,candidates)

        # apply end systems
        candidates_ranked_sim_cutoff = candidates_ranked_sim[:5]
        candidates_ranked_sim_filtered = [c for c in candidates_ranked_sim if c[-1] == 0]
        if model == 'pack':
            candidates_ranked_sim_filtered_cutoff = candidates_ranked_sim_filtered[:5]
        
        # print(len(candidates_ranked_sim_filtered))
        try:
            candidates_first_topic = self.rank_questions_topics(question,[prominent_topics[0]],candidates_ranked_sim_filtered)[0]
            candidates_first_topic_ranked = sorted(candidates_first_topic,key=lambda k : (k[-1],k[3]),reverse=True)[:5]
        # candidates_first_topic1_ranked_diverse = self.diversify(candidates_first_topic1_ranked,0.7)[:5]
        # candidates_first_topic2 = self.rank_questions_topics2(question,[prominent_topics[0]],candidates_ranked_sim_filtered)[0]
        # candidates_first_topic2_ranked = sorted(candidates_first_topic2,key=lambda k : (k[-1],k[3]),reverse=True)
        # # candidates_first_topic2_ranked_diverse = self.diversify(candidates_first_topic2_ranked,0.7)[:5]

        # if len(prominent_topics) > 1:
            # print('RANK BY ALL PROMINENT TOPICS')
            candidates_prominent_topics = [candidates_first_topic_ranked] + self.rank_questions_topics(question,prominent_topics[1:],candidates_ranked_sim_filtered) + [candidates_ranked_sim_filtered]
            ranking_topic = prominent_topics + ['plain_sim']
            candidates_prominent_topics_combined = []
            while len(candidates_prominent_topics_combined) < 5:
                c = len(candidates_prominent_topics_combined)
                for i,ranking in enumerate(candidates_prominent_topics):
                    if len(ranking) == 0:
                        continue
                    if ranking[0] == 0.0:
                        continue
                    candidates_ids = [x[0] for x in candidates_prominent_topics_combined]
                    for x in ranking:
                        if x[0] not in candidates_ids:
                            if i == len(candidates_prominent_topics)-1:
                                x.extend(['plain_sim',0])
                            candidates_prominent_topics_combined.append(x) 
                            break
                            # diverse = True
                            # for y in candidates_prominent_topics_combined:
                            #     softcosine = self.sim_model.softcos(y[1],y[2],x[1],x[2])
                            #     if softcosine > 0.7:
                            #         diverse = False
                            # if diverse:
                            #     candidates_prominent_topics_combined.append(x)
                            #     break
                if c == len(candidates_prominent_topics_combined): # no improvement
                    break
        # else:
        #     candidates_prominent_topics_combined = candidates_first_topic_ranked
            candidates_prominent_topics_ranked = sorted(candidates_prominent_topics_combined,key=lambda k : (k[-1],k[3]),reverse=True)[:5] 
        # ranked_candidates_prominent_topics_pop = candidates_first_topic2_ranked_diverse
        
        # return prominent_topics, candidates_ranked_sim, candidates_ranked_sim_cutoff, candidates_ranked_sim_filtered_cutoff, candidates_first_topic1_ranked_diverse, ranked_candidates_prominent_topics_pop
        except:
            candidates_prominent_topics_ranked = []
        if model == 'pack':
            return prominent_topics, candidates_ranked_sim, candidates_ranked_sim_cutoff, candidates_ranked_sim_filtered_cutoff, candidates_first_topic_ranked, candidates_prominent_topics_ranked
        else:
            return candidates_prominent_topics_ranked
