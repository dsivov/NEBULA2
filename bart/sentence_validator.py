from transformers import BartTokenizer, BartForConditionalGeneration
import numpy as np

class BART:
    def __init__(self):
        self.tokenizer = tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
        self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")


    def generate_score_on_mask(self, input_txt: str, to_print: bool) -> float:
        words = input_txt.split(" ")
        mask_token = '<mask>'
        cur_sentence = f'{words[0]} {mask_token}'
        cond_probs = [1]
        
        for word in words[1:]:
            input_ids = self.tokenizer([cur_sentence], return_tensors="pt")["input_ids"]
            logits = self.model(input_ids).logits
            masked_index = (input_ids[0] == self.tokenizer.mask_token_id).nonzero().item()
            probs = logits[0, masked_index].softmax(dim=0)
            idx_token = self.tokenizer.convert_tokens_to_ids([word])
            cond_probs.append(float(probs[idx_token][0]))

            cur_sentence = cur_sentence.replace(mask_token, '')
            cur_sentence += f'{word} {mask_token}'
        
        result = np.prod(cond_probs)
        if to_print:
            print(f'Input Text: {input_txt} , Result: {result}')
        return result
    
    def generate_score_on_str(self, input_txt: str, to_print: bool) -> float:
        words = input_txt.split(" ")
        mask_token = '<mask>'
        cond_probs = []
        
        for idx, word in enumerate(words):
            cur_sentence = input_txt.split(" ")
            cur_sentence[idx] = mask_token
            cur_sentence = ' '.join(cur_sentence)
            input_ids = self.tokenizer([cur_sentence], return_tensors="pt")["input_ids"]
            logits = self.model(input_ids).logits
            masked_index = (input_ids[0] == self.tokenizer.mask_token_id).nonzero().item()
            probs = logits[0, masked_index].softmax(dim=0)
            idx_token = self.tokenizer.convert_tokens_to_ids([word])
            cond_probs.append(float(probs[idx_token][0]))

        
        result = np.prod(cond_probs)
        norm_result = result**(1./len(input_txt.split(" ")))
        if to_print:
            print(f'Input Text: {input_txt} , Result: {norm_result}')
        return result


if __name__ == "__main__":
    input_txt_valid = 'white v-neck shirt'

    with open('./bart/sentence_correctness_classifier/annotated_sentences.txt') as f:
        input_txt = [line.rstrip('\n').rstrip().lstrip().replace(',', '').replace('\'','').replace('\xa0', '') for line in f]
        input_txt = [x for x in input_txt if x != '']

        bart = BART()
        print("Scoring when all the sentence is given and mask is moved to right: ")
        for input in input_txt:
            bart.generate_score_on_str(input, to_print=True)

    # print("Scoring when only the previous word are given: ")
    # bart.generate_score_on_mask(input_txt_valid, to_print=True)
    # bart.generate_score_on_mask(input_txt_invalid, to_print=True)



    # tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    # TXT = "My friends are <mask> but they eat too many carbs."

    # model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
    # input_ids = tokenizer([TXT], return_tensors="pt")["input_ids"]
    # logits = model(input_ids).logits

    # masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
    # probs = logits[0, masked_index].softmax(dim=0)
    # values, predictions = probs.topk(5)
    # tokenizer.decode(predictions).split()