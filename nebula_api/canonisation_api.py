from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer

#Please use the NLTK Downloader to obtain the resource:

import nltk
#  >>> nltk.download('wordnet')

class CANON_API:
    def __init__(self) -> None:
        pass
    
    def get_score(self, concept):
        #print(wordnet.synsets(concept))
        if len(wordnet.synsets(concept)) > 1:
            syn = wordnet.synsets(concept)[0]
            root = syn.root_hypernyms()[0]
            score = syn.path_similarity(root)
            return(score)
        else:
            return(1)

    def get_concept_from_entity(self, concept):
        nouns = self.get_noun_from_concept(concept)
        verbs = self.get_verb_from_concept(concept)
        return(nouns, verbs)
    
    def get_noun_from_concept(self, concept):
        ascore = self.get_score(concept) 
        class_ = self.get_class_of_entity(concept)
        concept_presesnt = WordNetLemmatizer().lemmatize(concept,'n')
        lem = wordnet.lemmas(concept_presesnt, pos='n')
        #print(lem)
        all_concepts = []
        if len(concept) > 2:
            related_forms = [lem[i].synset() for i in range(len(lem))]
            #print(related_forms)
            for related_form in related_forms:
                all_concepts.append(related_form.lemmas()[0].name())
                if ascore < 0.15 and class_ == 'person':
                    nn = related_form.hyponyms()
                    for rel in nn:
                        for r in rel.lemmas():
                            all_concepts.append(r.name().lower())
                # for rel in related_form.hyponyms():
                #     all_concepts.append(rel.lemmas()[0].name())  
            all_concepts = list(dict.fromkeys(all_concepts)) 
        return(all_concepts)

    def get_verb_from_concept(self, concept):
        concept_presesnt = WordNetLemmatizer().lemmatize(concept,'v')
        lem = wordnet.lemmas(concept_presesnt, pos='v')
        all_relations = []
        related_forms = [lem[i].synset() for i in range(len(lem))]
        for related_form in related_forms:
            #print("Related form ", related_form)
            all_relations.append(related_form.lemmas()[0].name().lower())
            # for rel in related_form.hypernyms():
            #     all_relations.append(rel.lemmas()[0].name())
            # for rel in related_form.hyponyms():
            #     all_relations.append(rel.lemmas()[0].name())
        all_relations = list(dict.fromkeys(all_relations))
        return(all_relations)

            # for form in related_form:
            #     print(form)
    def get_class_from_context(self, context):
        corpa = nltk.word_tokenize(context)
        concepts_pos = nltk.pos_tag(corpa) 
        for concept_pos in concepts_pos:
            print(concept_pos)


    def get_class_of_entity(self, concept):
        if len(wordnet.synsets(concept)) > 0:
            root = wordnet.synsets(concept)[0].root_hypernyms()[0].name()
            for syn in wordnet.synsets(concept):
                #syn = wordnet.synsets(concept)[0]
                #print(wordnet.synsets(concept))
               
                if syn.name() == root:
                    return("attribute")
                if len(syn.hypernyms()) >= 1:
                    abstract = syn.hypernyms()[0]
                else:
                    return('abstraction')
                while abstract.name() != root:
                    #print(syn.name())
                    if syn.name() == 'location.n.01' or syn.name() == 'area.n.05' or \
                        syn.name() == 'room.n.01' or syn.name() == 'road.n.01' or syn.name() == 'forest.n.02':
                        return('location')
                    elif syn.name() == 'artifact.n.01':
                        return('artifact')
                    elif syn.name() == 'person.n.01':
                        return('person')
                    elif syn.name() == 'body_part.n.01':
                        return('body_part')
                    elif syn.name() == 'social_group.n.01':
                        return('social_group')
                    elif syn.name() == 'animal.n.01':
                        return('animal')
                    elif syn.name() == 'plant.n.02':
                        return('plants')
                    elif syn.name() == 'attribute.n.02' or syn.name() == 'number.n.02':
                        return('attribute')
                    elif syn.name() == 'abstraction.n.06':
                        return('abstraction')
                    elif syn.name() == 'action.n.01' or syn.name() == 'behavior.n.01' or \
                        syn.name() == 'event.n.01' or syn.name() == 'cognition.n.01':
                        return('action')
                    elif syn.name() == 'emotion.n.01':
                        return('emotion')
                    else:
                        if len(abstract.hypernyms()) > 0:
                            syn = abstract
                            abstract = syn.hypernyms()[0]
                        else:
                            return('none')
                    #continue 
                # #path.append(syn)
            return('none')
        else:
            return('none')

    def get_path_to_root(self, concept):
        path = []
        if len(wordnet.synsets(concept)) > 0:
            syn = wordnet.synsets(concept)[0]
            root = syn.root_hypernyms()[0].name()
            if syn.name() == root:
                return(1)
            abstract = syn.hypernyms()[0]
            path.append(syn)
            path.append(abstract)
            while abstract.name() != root:
                syn = abstract
                abstract = syn.hypernyms()[0]
                #path.append(syn)
                path.append(abstract)
            return(path)
        else:
            return(1)

    def get_hyper_concepts(self, concept):
        syn = wordnet.synsets(concept)[0]
        hyper = syn.hypernyms()
        print(hyper)
        hypo = syn.hyponyms()
        print(hypo)
        holo = syn.member_holonyms()
        print(holo)

def main():
    while True:
       
        ascore = CANON_API()
        #score = ascore.get_hyper_concepts(concept)
        #path = ascore.get_path_to_root(concept)
        #print("ABSTRACTION SCORE: ", score)
        #print("FULL ABSTRACTION PATH: ",  path)
       
        
        concept = input("Concept> ")
        #print(ascore.get_concept_from_entity(concept))
        print(ascore.get_class_of_entity(concept))
if __name__ == '__main__':
    main()