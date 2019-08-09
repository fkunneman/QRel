
from qrel.classes import question

class QuestionRelator:
    """
    Class to retrieve a set of related questions to a given question, optimized for relevance, novelty and diversity
    """

    def __init__(self,sim_model):
        # A trained similarity model is essential to the question relatedness procedure
        self.sim_model = sim_model

    ##################################
    ### TOPIC FUNCTIONS ##############
    ##################################

    def select_topics(self,q,percentage=0.70):
        """
        Function to select the most prominent topics of a question
        The function assumes that topics have been extracted from a question, and ranked based on a score for their prominence
        With this function, the most prominent topics are selected based on a percentage:
            The most prominent topics of which the combined scores make up for over the given percentage of all scores combined are selected 
        """
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
        """
        Function to format a topic like a question object, so it can be used to score the similarity of questions to the topic
        """
        topic_tokens = topic['topic'].split()
        topic_obj = question.Question()
        topic_obj.set_tokens(topic_tokens)
        topic_obj.set_emb(self.sim_model.encode(topic_tokens))        
        return topic_obj

    ##################################
    ### SIMILARITY FUNCTIONS #########
    ##################################

    def retrieve_diverse_candidates(self,question,topic_cutoff,ntargets):
        """
        Function to retrieve a diverse set of candidate related questions based on BM25
        In order to retrieve questions that might link more to particular topics of a question and still adhere to
            the formulation in the given question, BM25 is requested to retrieve candidates for multiple variants:
            * the complete question
            * the question stripped from topic1
            * the question stripped frop topic2
            * etc.

        """
        if topic_cutoff == 1: # if question has only one prominent topic, only candidates based on the complete question are retrieved
            ntargets_by_chunk = ntargets
        else:
            ntargets_by_chunk = int(ntargets / (topic_cutoff+1)) # to make sure that the requested number of targets are retrieved, the size of the chunks per query question are decided here
        candidates = self.sim_model.retrieve_candidates(question.tokens,ntargets_by_chunk) # first retrieve candidates from the complete question
        cids = [question.id] + [c.id for c in candidates]
        if topic_cutoff > 1: 
            # for each topic (if more than 1)
            for i,topic in enumerate(question.topics[:topic_cutoff]):
                # retrieve questions based on the question minus the topic, and add to combined list (making sure that any retrieved question was not already in the list)
                candidates.extend([c for c in self.sim_model.retrieve_candidates(list(set(question.tokens) - set(topic['topic_text'].split())),ntargets_by_chunk) if c.id not in cids])
        return candidates

    def rank_questions_topic(self,question,topic,targets):
        """
        Function to rank questions based on their similarity to the given topic 
        """
        topic_obj = self.format_topic(topic) # format topic like a question object, so it can be used in the sim_model
        topic_ranked = self.sim_model.rerank_candidates(topic_obj,targets,'softcosine')
        return topic_ranked

    def rank_questions_topics(self,question,topic_cutoff,targets):
        """
        Function to rank questions based on their similarity to each topic
        """
        ranked_by_topic = []
        for topic in question.topics[:topic_cutoff]:
            ranked_by_topic.append(self.rank_questions_topic(question,topic,targets))
        return ranked_by_topic
    
    ##################################
    ### MAIN FUNCTIONS ###############
    ##################################

    def select_related_questions(self,question,topic_cutoff,candidates,n):
        """
        Function to select related questions to a given question based on ranked lists of the candidates' similarity to the question topics
            and the question as a whole
        To come to a diverse set of related questions, a mixture of the ranked list is compiled
        """
        topics_ranked = self.rank_questions_topics(question,topic_cutoff,[c[0] for c in candidates]) # rank candidate similarity to each topic
        topics_ranked_plus_sim = topics_ranked + [[c[:2] for c in candidates]] # add ranking based on general similarity to question
        related_questions = []
        topics = [t['topic'] for t in question.topics[:topic_cutoff]] + ['plain_sim']
        while len(related_questions) < n: # Continue until the target number of related questions is reached
            c = len(related_questions) 
            for i,ranking in enumerate(topics_ranked_plus_sim): # for each ranking of the candidates (by similarity to topics and to question as a whole)
                if len(ranking) == 0: # no candidates returned if there is no content in the ranking
                    continue
                if ranking[0] == 0.0: # no candidates returned if candidate is not similar at all
                    continue
                candidates_ids = [rq[0] for rq in related_questions] 
                for x in ranking:
                    if x[0].id not in candidates_ids: # make sure that question is not in related questions yet
                        related_questions.append([x[0].id,x[0].questiontext,x[1],topics[i]]) 
                        break
            if c == len(related_questions): # way to prevent infinite loop in case of no improvement
                break
        return related_questions[:5]

    def relate_question(self,question,topic_percentage=0.70,ncandidates=50,num_related=5):
        """
        Complete question relatedness procedure
        """
        # select prominent topics of question
        topic_cutoff = self.select_topics(question,topic_percentage)

        # retrieve candidate questions
        candidates = self.retrieve_diverse_candidates(question,topic_cutoff,ncandidates)
        candidates_output = [c.id for c in candidates]

        # score and rank questions by similarity
        ranked_candidates = self.sim_model.rerank_candidates(question,candidates,'ensemble')
        
        # filter questions labeled as similar (duplicate) by the similarity ranking
        ranked_candidates_filtered = [c for c in ranked_candidates if c[-1] == 0]

        # make a selection of five relevant, novel and diverse questions
        related_questions = self.select_related_questions(question,topic_cutoff,ranked_candidates_filtered,num_related)

        return related_questions, candidates_output
