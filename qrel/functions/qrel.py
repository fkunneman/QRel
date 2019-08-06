
from qrel.classes import question

class QuestionRelator:

    def __init__(self,sim_model):
        self.sim_model = sim_model

    ##################################
    ### TOPIC FUNCTIONS ##############
    ##################################

    def select_topics(self,q,percentage=0.70):
        total = sum([t['topic_score'] for t in q.topics])
        if total == 0:
            return []
        acc = 0
        cutoff = 0
        for topic in q.topics:
            cutoff += 1
            acc += topic['topic_score']
            if (acc/total > percentage):
                break
        return cutoff
       
    def format_topic(self,topic):
        topic_tokens = topic['topic'].split()
        topic_obj = question.Question()
        topic_obj.set_tokens(topic_tokens)
        topic_obj.set_emb(self.sim_model.encode(topic_tokens))        
        return topic_obj

    ##################################
    ### SIMILARITY FUNCTIONS #########
    ##################################

    def retrieve_diverse_candidates(self,question,topic_cutoff,ntargets):
        if topic_cutoff == 1:
            ntargets_by_chunk = ntargets
        else:
            ntargets_by_chunk = int(ntargets / (topic_cutoff+1))
        candidates = self.sim_model.retrieve_candidates(question.tokens,ntargets_by_chunk)
        cids = [question.id] + [c.id for c in candidates]
        if topic_cutoff > 1:
            for i,topic in enumerate(question.topics[:topic_cutoff]):
                candidates.extend([c for c in self.sim_model.retrieve_candidates(list(set(question.tokens) - set(topic['topic_text'].split())),ntargets_by_chunk) if c.id not in cids])
        return candidates

    def rank_questions_topic(self,question,topic,targets):
        topic_obj = self.format_topic(topic)
        topic_ranked = self.sim_model.rerank_candidates(topic_obj,targets,'softcosine')
        return topic_ranked

    def rank_questions_topics(self,question,topic_cutoff,targets):
        ranked_by_topic = []
        for topic in question.topics[:topic_cutoff]:
            ranked_by_topic.append(self.rank_questions_topic(question,topic,candidates))
    
    ##################################
    ### MAIN FUNCTIONS ###############
    ##################################

    def select_related_questions(question,topic_cutoff,candidates,n):
        topics_ranked = self.rank_questions_topics(question,topic_cutoff,[c[0] for c in candidates])
        topics_ranked_plus_sim = topics_ranked + [candidates]
        related_questions = []
        topics = [t['topic'] for t in question.topics[:topic_cutoff]] + ['plain_sim']
        while len(related_questions) < n:
            c = len(related_questions)
            for i,ranking in enumerate(topics_ranked_plus_sim):
                if len(ranking) == 0: # no candidates returned
                    continue
                if ranking[0] == 0.0: # candidate not similar
                    continue
                candidates_ids = [rq[0].id for rq in related_questions]
                for x in ranking:
                    if x[0].id not in candidates_ids: # make sure that question is not in related questions yet
                        related_questions.append([x,topics[i]]) 
                        break
            if c == len(related_questions): # no improvement
                break
        return related_questions

    def relate_question(self,question,topic_percentage=0.70,ncandidates=50,num_related=5):

        # select prominent topics
        topic_cutoff = self.select_topics(question,topic_percentage)

        # retrieve candidate_questions
        candidates = self.retrieve_diverse_candidates(question,topic_cutoff,ncandidates)

        # score and rank questions by similarity
        ranked_candidates = self.sim_model.rerank_candidates(question,candidates,'ensemble')
        
        # filter questions labeled as similar (duplicate) by the similarity ranking
        ranked_candidates_filtered = [c for c in ranked_candidates if c[-1] == 0]

        # make a selection of five relevant, novel and diverse questions
        related_questions = self.select_related_questions(question,topic_cutoff,ranked_candidates_filtered,num_related)

        return related_questions
