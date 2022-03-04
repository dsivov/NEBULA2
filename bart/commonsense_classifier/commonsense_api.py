from transformers import RobertaForSequenceClassification, RobertaTokenizerFast
import torch
from typing import List

class CS_API:
    def __init__(self,
                model_params_path='./bart/commonsense_classifier/Subtask_A_Best_Model_.pth',
                device_type='cpu',
                device_id='cuda:0'):
        if not torch.cuda.is_available() and device_type=='gpu':
            print("ERROR: GPU is not available in your system, please use `device_type=cpu`")
            return None
        if device_type=='cpu':
            self.device = 'cpu'
        else:
            self.device = torch.device(device_id if torch.cuda.is_available() else 'cpu')
        self.model_name = 'roberta-large'
        self.model_params_path = model_params_path
        print(f"Loading the model {self.model_name}...")
        self.model = RobertaForSequenceClassification.from_pretrained(self.model_name).to(self.device)
        self.tokenizer = RobertaTokenizerFast.from_pretrained(self.model_name)
        # Load the best-trained model
        self.model.load_state_dict(torch.load(model_params_path, map_location=torch.device(self.device)))
        print(f"The device your are using: {self.device}")
        print("Loaded model successfully.")
        print("You can use the API now for testing purposes.")

    def get_sentences_score(self, sentences : List[str]):
        """
        Get score for every sentence.
        """
        print("Starting to evaluate the input for the model..")
        output_preds = []
        for statement in sentences:
            # Tokenization for sentence
            inputs = self.tokenizer(statement, truncation=True, padding=True, return_tensors="pt").to(self.device)

            # Model makes its prediction
            outputs = self.model(**inputs)

            probabilities = torch.softmax(outputs["logits"], dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            if predictions == 1:
                prediction = float(probabilities[0][1])
            else:
                prediction = 1 - float(probabilities[0][0])

            output_preds.append(prediction)
        
        return output_preds

def main():
    weights_path = "./bart/commonsense_classifier/Subtask_A_Best_Model_.pth"
    cs_validator = CS_API(model_params_path=weights_path, device_type='gpu', device_id='cuda:0')
    test = ['hello', 'how are you?', 'People who look at the books and booklets take ties.']
    scores = cs_validator.get_sentences_score(test)
    print(scores)


if __name__ == "__main__":
    main()