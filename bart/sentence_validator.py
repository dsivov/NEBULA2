from transformers import BartTokenizer, BartForConditionalGeneration
import numpy as np

class BART:
    def __init__(self):
        self.tokenizer = tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
        self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")


    def generate_score_on_str(self, input_txt: str, to_print: bool) -> float:
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


if __name__ == "__main__":
    input_txt_valid = 'man sit on a chair'
    input_txt_invalid = 'chair sit on a man'

    bart = BART()
    bart.generate_score_on_str(input_txt_valid, to_print=True)
    bart.generate_score_on_str(input_txt_invalid, to_print=True)


    # tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    # TXT = "My friends are <mask> but they eat too many carbs."

    # model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
    # input_ids = tokenizer([TXT], return_tensors="pt")["input_ids"]
    # logits = model(input_ids).logits

    # masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
    # probs = logits[0, masked_index].softmax(dim=0)
    # values, predictions = probs.topk(5)
    # tokenizer.decode(predictions).split()