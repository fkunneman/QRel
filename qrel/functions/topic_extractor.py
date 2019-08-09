
import copy
import numpy

class TopicExtractor:
    """
    Class to extract topic segments from a given (Dutch) text, 
    using pre-trained models from Wikipedia and the GoeieVraag.nl categories
    """
    def __init__(self,ngram_commonness,ngram_entropy):
        # initialize with pretrained models (e.g.: lists with topic segments and their prominence score)
        print('Initializing commonness')
        self.set_commonness(ngram_commonness)
        print('Initializing entropy')
        self.set_entropy(ngram_entropy)

    ############
    ### INIT ###
    ############

    def set_commonness(self,ngram_commonness):
        """
        Initialize commonness scores of topic segments - titles of wikipedia pages and their commonness score
        The score is based on how often (relatively) the title of the page is used as a hyperlink from other pages
        """
        self.cs = {} # dictionary with ngram string as key and commonness score as value
        with open(ngram_commonness,'r',encoding='utf-8') as file_in:
            lines = file_in.read().strip().split('\n')
            for line in lines:
                tokens = line.split('\t')
                entity = tokens[0]
                self.cs[entity] = float(tokens[-1])
        self.commonness_set = set(self.cs.keys())

    def set_entropy(self,ngram_entropy):
        """
        Initialize entropy scores of topic segments - 
        the link between lemma ngrams in questions posed on goeievraag.nl with particular question categories 
        --> the more an ngram is used across different categories, the higher its entropy, and the lower its prominence as a topic ngram
        """
        self.entropy = {} # dictionary with ngram string as key and inverted entropy score as value 
        with open(ngram_entropy,'r',encoding='utf-8') as file_in:
            lines = file_in.read().strip().split('\n')
            for line in lines:
                tokens = line.split()
                entity = ' '.join(tokens[:-1])
                self.entropy[entity] = 1 - float(tokens[-1])
        self.entropy_set = set(self.entropy.keys())

    ###############
    ### HELPERS ###
    ###############

    def match_index(self,token,sequence1,sequence2):
        # function for returning the (stored) part-of-speech tag of a question
        # token is a word or lemma token, sequence1 is a list of lemmas or tokens, sequence2 is a list of postags
        index = sequence1.index(token)
        return sequence2[index]

    def filter_entities(self,entities,question):
        # function to remove topic segments that are likely uninformative
        # topic segments that are no Noun, Verb, Adverb or Adjective are arguably too uninformative
        filtered = []
        for entity in entities:
            tokens = entity.split()
            if len(tokens) > 1: # topic segments with multiple tokens are often informative 
                filtered.append(entity)
            else:
                entity = tokens[0]
                if len(entity) <= 1:
                    continue
                try:
                    pos = self.match_index(entity,question.lemmas,question.pos)
                    if not pos in ['DET','PRON','ADP','ADV','CCONJ','SCONJ','CONJ']: # check for postag
                        filtered.append(entity)
                except:
                    print('COULD NOT FIND INDEX FOR',entity.encode('utf-8'),'in',' '.join(question.lemmas).encode('utf-8'))
                    continue
        return filtered

    def rerank_topics(self,topics_commonness,topics_entropy):
        """ 
        Function to combine commonness and entropy scores for topic segments into a single score
        The two metrics differ in which topic segments they include and how much weight those are given


        """
        topics_commonness_txt = [x[0] for x in topics_commonness]
        topics_entropy_txt = [x[0] for x in topics_entropy]
        topics_commonness_only = list(set(topics_commonness_txt) - set(topics_entropy_txt))
        topics_entropy_only = list(set(topics_entropy_txt) - set(topics_commonness_txt))
        topics_union = list(set(topics_commonness_txt).union(set(topics_entropy_txt)))
        topics_commonness_complete = copy.deepcopy(topics_commonness)
        for topic in topics_entropy_only:
            topics_commonness_complete.append([topic,0]) # the entropy topics that are not scored with commonness are appended with '0' for commonness 
        topics_entropy_complete = copy.deepcopy(topics_entropy)
        for topic in topics_commonness_only:
            topics_entropy_complete.append([topic,0]) # the commonness topics that are not scored with entropy are appended with '0' for entropy
        topics_commonness_complete_txt = [x[0] for x in topics_commonness_complete]
        topics_entropy_complete_txt = [x[0] for x in topics_entropy_complete]
        topics_combined = []
        for topic in topics_union:
            score_commonness = topics_commonness_complete[topics_commonness_complete_txt.index(topic)][1]
            score_entropy = topics_entropy_complete[topics_entropy_complete_txt.index(topic)][1]
            avg = numpy.mean([score_commonness,score_entropy]) # the topics that are scored by both metrics are scored as the average of the two
            topics_combined.append([topic,avg,score_entropy,score_commonness])
        topics_ranked = sorted(topics_combined,key = lambda k : k[1],reverse=True) # rank topics by combined score
        return topics_ranked

    def reduce_overlap(self,ranked_topics):
        """
        Function to reduce overlap in topics segments
        Some topics segments are redundant to each other when they partly overlap
        In this case the topic segment with the lower score of the two is removed
        """
        filtered_topics = []
        for topic in ranked_topics:
            overlap = False
            for j,placed_topic in enumerate(filtered_topics):
                if set(topic[0].split()) & set(placed_topic[0].split()):
                    overlap = True
                    break
            if not overlap:
                filtered_topics.append(topic)
        return filtered_topics

    def topic2text(self,topics,question):
        """
        Topic segments are extracted as a sequence of lemma's, 
        which is not insightful to present the particular topic segment for a given question
        This function returns the actual words of the topic in the question as a string, 
        based on the position of the lemmas in the question
        """
        topics_text = []
        for topic in topics:
            tokens = topic.split()
            if len(tokens) > 1:
                startindices = []
                for i in range(len(tokens)):
                    indices = [j for j,x in enumerate(question.lemmas) if x == tokens[i]]
                    if len(startindices) == 0:
                        startindices = indices
                    else:
                        new_startindices = []
                        for index in indices:
                            for si in startindices:
                                if index == si+1:
                                    new_startindices.append(index)
                        if len(new_startindices) == 0:
                            print("could not find indices for",topic,sequence)
                        startindices = new_startindices
                if not len(startindices) == 1:
                    print("No single start index for",topic.encode('utf-8'),' '.join([x[0] for x in question.lemmas]).encode('utf-8'),"startindex",startindices)
                index = startindices[0] - len(tokens)
                text = ' '.join(question.tokens[startindices[0]-1:startindices[0]+(len(tokens)-1)])
            else:
                entity = tokens[0]
                text = question.tokens[question.lemmas.index(entity)]
            topics_text.append(text)
        return topics_text
    
    ###############
    ### EXTRACT ###
    ###############

    def extract(self,question,max_topics=5):
        """
        Function to apply all steps of topic extraction to a given question, and return a ranked list of:
        - extracted topic (sequence of lemmas)
        - score of topic
        - particular commonness score of topic
        - particular entropy score of topic
        - topic as it occurs in the given question
        """
        topics_commonness = [[e,self.cs[e]] for e in self.filter_entities(list(set(question.lemmas) & self.commonness_set),question) if self.cs[e] > 0.05]
        topics_entropy = [[e,self.entropy[e]] for e in self.filter_entities(list(set(question.lemmas) & self.entropy_set),question)]
        topics_ranked = self.rerank_topics(topics_commonness,topics_entropy)
        topics_filtered = self.reduce_overlap(topics_ranked)[:max_topics]
        topics_text = self.topic2text([x[0] for x in topics_filtered],question)
        topics_filtered_text = [tf + [topics_text[i]] for i,tf in enumerate(topics_filtered)]
        topics_filtered_text_dict = [{'topic':x[0],'topic_score':x[1],'topic_entropy':x[2],'topic_commonness':x[3],'topic_text':x[4]} for x in topics_filtered_text]
        return topics_filtered_text_dict

    def extract_list(self,questions,max_topics=5):
        # Function to extract topics from multiple questions
        return [self.extract(q,max_topics) for q in questions]
