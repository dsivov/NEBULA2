import json
import torch
import re
import argparse
from tqdm import tqdm
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from nltk.corpus import words
from nebula_api.atomic2020.utils import calculate_rouge, use_task_specific_params, calculate_bleu_score, trim_batch


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class Comet:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.model.zero_grad()
        print("model loaded")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        task = "summarization"
        use_task_specific_params(self.model, task)
        self.batch_size = 1
        self.decoder_start_token_id = None
        self.triplet_relations = [
            "DefinedAs",
            "HasA",
            "xReason",
            "HasFirstSubevent",
            "HasLastSubevent", 
            "HasSubEvent",
            "HasSubevent",
            "RelatedTo", 
            "isAfter",
            "isBefore" 
        ]
        self.person_relations = [
            "xEffect", 
            "xReact",
            "xWant",
            "xIntent",
            "xNeed"
            ]
        self.attributes_relations = [ 
            #"InstanceOf",
            #"MadeUpOf",
            "xAttr"
            ]
        self.comcepts_relations = [ 
            "isFilledBy"
            ]

    def generate(
            self, 
            queries,
            decode_method="beam", 
            num_generate=5, 
            ):

        with torch.no_grad():
            examples = queries

            decs = []
            for batch in list(chunks(examples, self.batch_size)):

                batch = self.tokenizer(batch, return_tensors="pt", truncation=True, padding="max_length").to(self.device)
                input_ids, attention_mask = trim_batch(**batch, pad_token_id=self.tokenizer.pad_token_id)
                #print(batch)
                summaries = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_start_token_id=self.decoder_start_token_id,
                    num_beams=num_generate,
                    num_return_sequences=num_generate,
                    )

                dec = self.tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                decs.append(dec)

            return decs
   
    def check_triplet(self, triplet):
        triplet = triplet.split(' ')
        if len(triplet) < 4:
            return(False)
        # for concept in triplet:
        #     print(concept)
        #     if concept in words.words() or concept == '':
        #         print("OK")
        #         continue
        #     else:
        #         return(False)
        return(True)
    
    def get_groundings(self, events, places=None, type='concepts', person='person'):
        if type == 'concepts':
            relations = self.comcepts_relations
            num_generate = 50
        elif type == 'attributes':
            relations = self.attributes_relations
            num_generate = 50
        elif type == 'triplet':
            relations = self.triplet_relations
            num_generate = 50
        elif type == 'person':
            relations = self.person_relations
            num_generate = 50
        else:
            print("Bad relation type: " + type)
        pplaces = []
        
        if places:
            for place in places:
                pplaces.append("person " + place)
        lighthouses = events + pplaces
        #print(lighthouses)
        groundings = []
        for lighthouse in lighthouses:
            lighthouse = re.sub("\d+", "person", lighthouse)
            #print(lighthouse)
            for rel in relations:
                queries = []  
                query = "{} {} [GEN]" .format(lighthouse, rel)
                queries.append(query)
                #print(rel)
                results = self.generate(queries, decode_method="beam", num_generate = num_generate)
                for result in results:
                    for grounding in result:
                        if type == 'person':
                            if rel == 'xNeed':
                                grounding = person + " need" + grounding
                            elif rel == 'xIntent':
                                grounding = person + " intent" + grounding
                            elif rel == 'xWant':
                                grounding = person + " want" + grounding
                            else:
                                grounding = person + grounding
                        if type == 'triplet':
                            if not self.check_triplet(grounding):
                               continue 
                        groundings.append(grounding)
        groundings = list(dict.fromkeys(groundings))
        return(groundings)
        

if __name__ == "__main__":
    print("model loading ...")
    # heads = ["1 hurriedly steps off of the curb into the street carrying her luggage to the car","3 passes by on her way home",
    # "the driver of the car has pulled up next to the woman"]
    # heads = ['2 is holding onto a mackeral and tossing his head back laughing with 1','4 growls at the start of the battle',
    # '2 laughs with the others','1 - 5 stand over a freshly slaughtered hog',
    # '3 places someone ashes in the sea','man stands with one on the beach','on a battlefield over the ocean','on a rocky beach']
    events = ['4 spots someone up ahead','5 and 7 work at the hotel and are waiting by the doors peering out the glass for someone',
    '3 eyes someone outside of the office','3 opens the door for the detectives with a calm demeanor','4 looks back as he crosses',
    'man quickly escorts someone down the lobby','the group move through the elevator']
    places = ['in a gatehouse of an airport','at an airport entrance','on the sstreet','outside the staff entrance']
    print("Original lighthouse---------------")
    print(events)
    comet = Comet("./comet-atomic_2020_BART")
    # comet = Comet("/home/migakol/deployment/data/comet-atomic_2020_BART")
    res = comet.get_groundings(events, places, 'concepts')
    print("Concepts.....")
    print(res)
    res = comet.get_groundings(events, places, 'attributes')
    print("Attributes...")
    print(res)
    res = comet.get_groundings(events, places, 'person','somebody')
    print("Persons, grounded by \"somebody\"")
    print(res)
    res = comet.get_groundings(events, places, 'triplet')
    print("Triplets")
    print(res)
    # results = results1[0] + results2[0]
    # results = list(dict.fromkeys(results))
    # for res in results:
    #     print(res)
    # input()
    
    
      