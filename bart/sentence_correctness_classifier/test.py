from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from typing import List
import torch.nn.functional as F

def ret_trained_model(path):

    model = BertForSequenceClassification.from_pretrained(
        path, 
        num_labels = 2, 
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
        return_dict=False
    )

    return model

def ret_dataloader(test_dataset, batch_size=1):

    test_dataloader = DataLoader(
                test_dataset, # The validation samples.
                sampler = SequentialSampler(test_dataset), # Pull out batches sequentially.
                batch_size = batch_size # Evaluate with this batch size.
            )
    return test_dataloader

def test_sentence(input_txt : str, weights_path : str):

    input_ids = []
    attention_masks = []
    label = 0
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    sentence = input_txt
    encoded_dict = tokenizer.encode_plus(
                        sentence,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 64,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                    )
        
    # Add the encoded sentence to the list.    
    input_ids.append(encoded_dict['input_ids'])

    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    label = torch.tensor(label)

    # Print input sentence, and Token IDs.
    # print('Original: ', sentence)
    # print('Token IDs:', input_ids)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = ret_trained_model(weights_path)
    model.to(device)

    input_ids = input_ids.to(device)
    attention_masks = attention_masks.to(device)
    label = label.to(device)

    (loss, logits) = model(input_ids, 
                        token_type_ids=None, 
                        attention_mask=attention_masks,
                        labels=label)

    logits = logits.detach().cpu().numpy()
    pred_flat = np.argmax(logits, axis=1).flatten()
    print(pred_flat)


def test_batch_sentences(input_txt : List[str], weights_path : str):
    input_ids = []
    attention_masks = []
    labels = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = ret_trained_model(weights_path)
    model.to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    sentences = input_txt

    for sent in sentences:

        encoded_dict = tokenizer.encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 64,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                    )
        
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
        
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

        labels.append(0)

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0).to(device)
    attention_masks = torch.cat(attention_masks, dim=0).to(device)
    labels = torch.tensor(labels).to(device)

    test_dataset = TensorDataset(input_ids, attention_masks, labels)

    validation_dataloader = ret_dataloader(test_dataset, batch_size=1)

    model = ret_trained_model(weights_path).to(device)
    # Logits
    preds = []
    # Softmax probability
    prob = []
    for batch in validation_dataloader:
                
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():        
            (loss, logits) = model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask,
                                    labels=b_labels)
            
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()

            # Add prediction of sentence(s)
            pred_flat = np.argmax(logits, axis=1).flatten()
            cur_prob = F.softmax(torch.tensor(logits), dim=1)
            
            if (float(max(cur_prob[0])) > 0.9):
                preds.append(pred_flat)
                prob.append(float(max(cur_prob[0])))
    
    print('========== RESULTS ==========\n')
    output = []
    print_table = [['ID', 'Input', 'Output', 'Probability']]
    for idx, pred in enumerate(preds):
        print_table.append([str(idx), input_txt[idx], str(pred), str(prob[idx])])

    length_list = [len(element) for row in print_table for element in row]
    column_width = max(length_list)
    for row in print_table:
        row = "".join(element.ljust(column_width + 1)  for element in row)
        output.append(row)
        # print(row)
    # np.savetxt('./bart/sentence_correctness_classifier/Output1.txt', output, fmt='%s')
    with open("./bart/sentence_correctness_classifier/Output_predictions.txt.txt", "w") as text_file:
        for row in print_table:
            row = "".join(element.ljust(column_width + 1)  for element in row)
            text_file.write(row + '\n')

    # for idx, pred in enumerate(preds):
    #     print("Sentence ")
    #     print(f'Sentence: {input_txt[idx]}, Prediction: {pred}')
        
if __name__ == "__main__":
    input_txt = 'chair on man'
    weights_path = "./bart/sentence_correctness_classifier/weights-20220215Feb02"
    
    # Using one sentence only
    # test_sentence(input_txt, weights_path)

    # input_batch_txt = [ 'chair on man',
    #                     'happy dressed up house',
    #                     'bearded face table',
    #                     'blue-eyed table',
    #                     'man sits on a chair',
    #                     'Dressed up men and women walking away in the evening',
    #                     'Elegantly dressed crowd walking away at night',
    #                     'A few men and women getting out of a wooden door in front of metal scaffolds',
    #                     'A few people getting out of the door in the morning',
    #                     'An elegant bunch of people standing',
    #                     'A group of men and women in a room',
    #                     'A soldier running away in a battlefield'
    #                 ]
    
    with open('./bart/sentence_correctness_classifier/annotated_sentences.txt') as f:
        input_batch_txt = [line.rstrip('\n').replace(',', '').replace('\'','').replace('\xa0', '') for line in f]
        test_batch_sentences(input_batch_txt, weights_path)