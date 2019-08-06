
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

class QuestionRelator:

    def __init__(self,sim_model):
        self.sim_model = sim_model         

    def return_prominent_topics(self,topics,percentage=0.70):
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

    def deduplicate_candidates(self,q_index,candidates):
        qids = []
        deduplicated = []
        for retrieved in bm25_output:
            rid = self.sim_model.idx2id[retrieved[0]]
            if rid != q_index:
                if rid not in qids:
                    qids.append(rid)
                    deduplicated.append(retrieved+[rid])
        return deduplicated
                    
    def retrieve_diverse_candidates(self,question,topics,ntargets):
        if len(topics) == 1:
            ntargets_by_chunk = ntargets
        else:
            ntargets_by_chunk = int(ntargets / (len(topics)+1))
        candidates = self.sim_model.retrieve_candidates(question.tokens,ntargets_by_chunk)
        if len(topics) > 1:
            for i,topic in enumerate(topics):
                candidates.extend(self.sim_model.retrieve_candidates(list(set(question.tokens) - set(topic['topic_text'].split())),ntargets_by_chunk))
        candidates_ranked = sorted(candidates_scores,key = lambda k : k[1],reverse=True)
        candidates_deduplicated = self.deduplicate_candidates(self.id2q[question.id],candidates_ranked)
        return scores_deduplicated

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
    
    def relate_question(self,question,topic_percentage=0.70,ncandidates=50):

        # select prominent topics
        prominent_topics = self.return_prominent_topics(question.topics,topic_percentage)

        # retrieve candidate_questions
        candidates = self.retrieve_diverse_candidates(question,prominent_topics,ncandidates)

        # score and rank questions by similarity
        ranked_candidates = self.sim_model.rerank_candidates(question,candidates)

        # apply end systems
        ranked_candidates_nondup = [c for c in ranked_candidates if c[-1] == 0]

        try:
            candidates_first_topic = self.rank_questions_topics(question,[prominent_topics[0]],ranked_candidates_nondup)[0]
            candidates_first_topic_ranked = sorted(candidates_first_topic,key=lambda k : (k[-1],k[3]),reverse=True)[:5]
            candidates_prominent_topics = [candidates_first_topic_ranked] + self.rank_questions_topics(question,prominent_topics[1:],ranked_candidates_nondup) + [ranked_candidates_nondup]
            ranking_topic = prominent_topics + ['plain_sim']
            related_questions = []
            while len(related_questions) < 5:
                c = len(related_questions)
                for i,ranking in enumerate(candidates_prominent_topics):
                    if len(ranking) == 0:
                        continue
                    if ranking[0] == 0.0:
                        continue
                    candidates_ids = [rq[0].id for rq in related_questions]
                    for x in ranking:
                        if x[0].id not in candidates_ids:
                            if i == len(candidates_prominent_topics)-1:
                                x.extend(['plain_sim',0])
                            related_questions.append(x) 
                            break
                if c == len(related_questions): # no improvement
                    break

            related_questions = sorted(related_questions,key=lambda k : (k[-1],k[3]),reverse=True)[:5] 
        except:
            print('Error when extracting related questions')
            related_questions = []
        return related_questions
