import pandas as pd
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW, BartConfig, BartModel, AutoTokenizer
import torch


def process_full_video():
    save_dir_annotation = '/home/migakol/data/small_lsmdc_test/'
    someone_annotation = '/home/migakol/data/small_lsmdc_test/gt/annotations-original.csv'
    # movies = get_dataset_movies()

    gt_data = pd.read_csv(someone_annotation, encoding='unicode_escape', delimiter='\t')
    # gt_data = gt_data['Her mind wanders for a beat.'].tolist()

    movie = '0033_Amadeus'
    movie_folder = '/dataset/lsmdc/avi/'

    # Go over all the frames of the movie and


def bart_enocder_decoder_examples():
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', add_prefix_space=True)
    bart_model = BartForConditionalGeneration.from_pretrained(
        "facebook/bart-base")
    # model = BartModel.from_pretrained("facebook/bart-large")
    model = BartModel.from_pretrained("facebook/bart-base")

    sentences = [['woman is at the kitchen', 'woman with tie is holding a plate'],
                 ['woman exists the door', 'she is leaving']]

    # Encode
    # When used with mask, decoder will return only the masked word
    # If <mask> is replaced with a word, decoder will decode everything
    inputs = tokenizer('I love <mask>', return_tensors='pt')
    aaa = model.encoder(**inputs)
    outputs = bart_model.generate(None, encoder_outputs=aaa, return_dict=True)[0]
    # logits = bart_model.generate(None, encoder_outputs=aaa, return_dict=True).logits
    tokenizer.decode(outputs, skip_special_tokens=True)


    num_beams = 2
    input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
    input_ids = input_ids * model.config.decoder_start_token_id
    encoder_input_ids = tokenizer('I love <mask>', return_tensors="pt").input_ids
    model_kwargs = {"encoder_outputs": model.get_encoder()(encoder_input_ids.repeat_interleave(num_beams, dim=0),
                                                            return_dict=True)}
    # out1 = bart_model.generate(None, encoder_outputs=aaa['encoder_last_hidden_state'], decoder_input_ids=decoder_input_ids,
    #                   return_dict=True).logits

    outputs = bart_model.generate(decoder_input_ids=input_ids, **model_kwargs)
    out_text = tokenizer.decode(outputs, skip_special_tokens=True)


if __name__ == '__main__':
    print('Started Mutli Scene Element Processing')
    bart_enocder_decoder_examples()
