from nltk.corpus import wordnet
#Please use the NLTK Downloader to obtain the resource:

#  >>> import nltk
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

    def get_verb_from_concept(self, concept):
        lem = wordnet.lemmas(concept, pos='v')
        all_relations = []
        related_forms = [lem[i].synset() for i in range(len(lem))]
        for related_form in related_forms:
            all_relations.append(related_form.lemmas()[0].name())
            for rel in related_form.hypernyms():
                all_relations.append(rel.lemmas()[0].name())
            for rel in related_form.hyponyms():
                all_relations.append(rel.lemmas()[0].name())
        all_relations = list(dict.fromkeys(all_relations))
        return(all_relations)

            # for form in related_form:
            #     print(form)

    def get_class_of_entity(self, concept):
        if len(wordnet.synsets(concept)) > 0:
            syn = wordnet.synsets(concept)[0]
            root = syn.root_hypernyms()[0].name()
            if syn.name() == root:
                return("attribute")
            abstract = syn.hypernyms()[0]
            while abstract.name() != root:
                #print(syn.name())
                if syn.name() == 'artifact.n.01':
                    return('artifact')
                elif syn.name() == 'person.n.01':
                    return('person')
                elif syn.name() == 'animal.n.01':
                    return('animal')
                elif syn.name() == 'abstraction.n.06':
                    return('abstraction')
                elif syn.name() == 'action.n.01':
                    return('action')
                elif syn.name() == 'emotion.n.01':
                    return('emotion')
                else:
                    syn = abstract
                    abstract = syn.hypernyms()[0]
                    #continue 
                # #path.append(syn)
            return('unknown')
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
        print(ascore.get_verb_from_concept(concept))

if __name__ == '__main__':
    main()