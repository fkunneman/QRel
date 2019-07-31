
import copy
import numpy

class TopicExtractor:

    def __init__(self,ngram_commonness,ngram_entropy):
        print('Initializing commonness')
        self.set_commonness(commonness_dir)
        print('Initializing entropy')
        self.set_entropy(ngram_entropy)

    ############
    ### INIT ###
    ############

    def set_commonness(self,ngram_commonness):
        self.cs = {}
        with open(ngram_commonness,'r',encoding='utf-8') as file_in:
            lines = file_in.read().strip().split('\n')
            for line in lines:
                tokens = line.split()
                entity = ' '.join(tokens[:-1])
                self.cs[entity] = float(tokens[-1])
        self.commonness_set = set(self.entropy.keys())
        
    def set_entropy(self,ngram_entropy):
        self.entropy = {}
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
        index = sequence1.index(token)
        return sequence2[index]
  
    def rerank_topics(self,topics_commonness,topics_entropy):
        topics_commonness_txt = [x[0] for x in topics_commonness]
        topics_entropy_txt = [x[0] for x in topics_entropy]
        topics_commonness_only = list(set(topics_commonness_txt) - set(topics_entropy_txt))
        topics_entropy_only = list(set(topics_entropy_txt) - set(topics_commonness_txt))
        topics_union = list(set(topics_commonness_txt).union(set(topics_entropy_txt)))
        topics_commonness_complete = copy.deepcopy(topics_commonness)
        for topic in topics_entropy_only:
            topics_commonness_complete.append([topic,0])
        topics_entropy_complete = copy.deepcopy(topics_entropy)
        for topic in topics_commonness_only:
            topics_entropy_complete.append([topic,0])
        topics_commonness_complete_txt = [x[0] for x in topics_commonness_complete]
        topics_entropy_complete_txt = [x[0] for x in topics_entropy_complete]
        topics_combined = []
        for topic in topics_union:
            score_commonness = topics_commonness_complete[topics_commonness_complete_txt.index(topic)][1]
            score_entropy = topics_entropy_complete[topics_entropy_complete_txt.index(topic)][1]
            avg = numpy.mean([score_commonness,score_entropy])
            topics_combined.append([topic,avg,score_entropy,score_commonness])
        topics_ranked = sorted(topics_combined,key = lambda k : k[1],reverse=True)
        return topics_ranked

    def reduce_overlap(self,ranked_topics):
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
                text = ' '.join(text_sequence[startindices[0]-1:startindices[0]+(len(tokens)-1)])
            else:
                entity = tokens[0]
                text = text_sequence[question.lemmas.index(entity)]
            topics_text.append(text)
        return topics_text
    
    ###############
    ### EXTRACT ###
    ###############

    def extract(self,question,max_topics=5):
        topics_commonness = [[e,self.cs[e]] for e in self.filter_entities(list(set(question.lemmas) & self.commonness_set),question)]
        topics_entropy = [[e,self.entropy[e]] for e in self.filter_entities(list(set(question.lemmas) & self.entropy_set),question)] 
        topics_ranked = self.rerank_topics(topics_commonness,topics_entropy)
        topics_text = self.topic2text([x[0] for x in topics_ranked],question)
        topics_filtered_text = [tf + [topics_text[i]] for i,tf in enumerate(topics_filtered)]
        topics_filtered_text_dict = [{'topic':x[0],'topic_score':x[1],'topic_entropy':x[2],'topic_commonness':x[3],'topic_text':x[4]} for x in topics_filtered_text]
        return topics_filtered_text_dict

    def extract_list(self,questions,max_topics=5):
        return [self.extract(q,max_topics) for q in questions]
