from nltk.corpus import wordnet

#Please use the NLTK Downloader to obtain the resource:

#  >>> import nltk
#  >>> nltk.download('wordnet')

class AS_API:
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

    def get_path_to_root(self, concept):
        path = []
      
        if len(wordnet.synsets(concept)) > 1:
            syn = wordnet.synsets(concept)[0]
            root = syn.root_hypernyms()[0].name()
            abstract = syn.hypernyms()[0]
            print(abstract)

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

def main():
    concept = input()
    ascore = AS_API()
    score = ascore.get_score(concept)
    path = ascore.get_path_to_root(concept)
    print("ABSTRACTION SCORE: ", score)
    print("FULL ABSTRACTION PATH: ",  path)

if __name__ == '__main__':
    main()