from gramformer import Gramformer
import torch
import pandas as pd
import numpy as np

def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(1212)

gf = Gramformer(models = 1, use_gpu=False) # 1=corrector, 2=detector

influent_sentences = [
    "He are moving here.",
    "I am doing fine. How is you?",
    "How is they?",
    "Matt like fish",
    "the collection of letters was original used by the ancient Romans",
    "We enjoys horror movies",
    "Anna and Mike is going skiing",
    "I walk to the store and I bought milk",
    " We all eat the fish and then made dessert",
    "I will eat fish for dinner and drink milk",
    "what be the reason for everyone leave the company",
]   

# Load the dataset into a pandas dataframe.
df = pd.read_csv("./bart/cola_dataset/raw/in_domain_train.tsv", delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])

# Display 10 random rows from the data.
df.sample(10)

# Get the lists of sentences and their labels.
sentences = df.sentence.values
labels = df.label.values

influent_sentences = [sentence for idx, sentence in enumerate(sentences) if labels[idx] == 0]

output = []
output_path = "./bart/grammer_checker/gram_results.txt"
NUM_OF_RESULTS = len(influent_sentences)
for influent_sentence in influent_sentences[:NUM_OF_RESULTS]:
    corrected_sentences = gf.correct(influent_sentence, max_candidates=1)
    output_result = f"[Input] {influent_sentence}\n[Correction] {corrected_sentences.pop()} \n {'-' * 100}"
    output.append(output_result)

np.savetxt(output_path, output, fmt='%s')