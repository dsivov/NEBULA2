import json
import torch
import argparse
from tqdm import tqdm
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from utils import calculate_rouge, use_task_specific_params, calculate_bleu_score, trim_batch


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class Comet:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        task = "summarization"
        use_task_specific_params(self.model, task)
        self.batch_size = 1
        self.decoder_start_token_id = None

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

visual_relations = [
    "DefinedAs",
    "HasA",
    "HasFirstSubevent",
    "HasLastSubevent", 
    "HasSubEvent",
    "HasSubevent",
    "RelatedTo", 
    "isAfter",
    "isBefore",   
    "xEffect",
    "xIntent",
    "xNeed",
    "xReact",
    "xReason",
    "xWant",
    ]

comcepts_relations = [ 
    "InstanceOf",
    "IsA",
    "MadeUpOf",
    "isFilledBy",
    "xAttr",
    ]

all_relations = [
    "AtLocation",
    "CapableOf",
    "Causes",
    "CausesDesire",
    "CreatedBy",
    "DefinedAs",
    "DesireOf",
    "Desires",
    "HasA",
    "HasFirstSubevent",
    "HasLastSubevent",
    "HasPainCharacter",
    "HasPainIntensity",
    "HasPrerequisite",
    "HasProperty",
    "HasSubEvent",
    "HasSubevent",
    "HinderedBy",
    "InheritsFrom",
    "InstanceOf",
    "IsA",
    "LocatedNear",
    "LocationOfAction",
    "MadeOf",
    "MadeUpOf",
    "MotivatedByGoal",
    "NotCapableOf",
    "NotDesires",
    "NotHasA",
    "NotHasProperty",
    "NotIsA",
    "NotMadeOf",
    "ObjectUse",
    "PartOf",
    "ReceivesAction",
    "RelatedTo",
    "SymbolOf",
    "UsedFor",
    "isAfter",
    "isBefore",
    "isFilledBy",
    "oEffect",
    "oReact",
    "oWant",
    "xAttr",
    "xEffect",
    "xIntent",
    "xNeed",
    "xReact",
    "xReason",
    "xWant",
    ]

if __name__ == "__main__":
    print("model loading ...")
    heads = ["person hurriedly steps off of the curb into the street carrying her luggage to the car on a narrow street or alley"]
    # comet1 = Comet("./comet-atomic_2020_BART_aaai")
    # comet1.model.zero_grad()
    comet2 = Comet("./comet-atomic_2020_BART")
    comet2.model.zero_grad()
    print("model loaded")
    for head in heads:
        # results1 = comet1.generate([head + " HasSubevent [GEN]"], decode_method="beam", num_generate=5)
        concepts1 = comet2.generate([head + " isFilledBy [GEN]"], decode_method="beam", num_generate=100)
        concepts2 = comet2.generate([head + " xReact [GEN]"], decode_method="beam", num_generate=100)
        concepts2 = comet2.generate([head + " xAttr [GEN]"], decode_method="beam", num_generate=100)
        concepts2 = comet2.generate([head + " InstanceOf [GEN]"], decode_method="beam", num_generate=100)

        #print(results1)
        print(results2)
    # results = results1[0] + results2[0]
    # results = list(dict.fromkeys(results))
    # for res in results:
    #     print(res)
    # input()
    t
    for rel in all_relations:
        queries = []  
        query = "{} {} [GEN]" .format(heads[0], rel)
        queries.append(query)
        print(rel)
        results = comet2.generate(queries, decode_method="beam", num_generate = 20)
        print(results)
        
      