from nebula_api.milvus_api import connect_db
from datasets import list_metrics, load_metric
import pathlib
import os
import pandas as pd
from bert_score import BERTScorer
import numpy as np

class CompareText:
    def __init__(self):
        self.curr_metric = None
        self.curr_name = None

    def compare_sentences(self, test_sent, gt_sent, metric='meteor'):
        """
        :param test_sent: test sentence, string
        :param gt_sent: ground truth sentence, string
        :return: P - precision, R - recall, S - method specific
            If only some values are available, the rest is None
            The return value differs according to the metric
            Meteor - it is a combination of recall and precison, with recall having much larger value
            Blue - precision
            Bert - precision, recall, F1
        """
        if self.curr_metric is None or self.curr_name != metric:
            if metric == 'bert':
                self.curr_metric = BERTScorer(lang="en", rescale_with_baseline=True)
            else:
                self.curr_metric = load_metric(metric)
            self.curr_name = metric
        if metric == 'meteor':
            scores = self.curr_metric.compute(predictions=[test_sent], references=[gt_sent])
            P = None
            R = None
            S = scores['meteor']
        elif metric == 'bleu':
            scores = self.curr_metric.compute(predictions=[test_sent.split(' ')], references=[[gt_sent.split(' ')]])
            P = scores['precisions'][0]
            R = None
            S = None
        elif metric == 'bert':
            P, R, S = self.curr_metric.score([test_sent], [gt_sent])
            P = 0.5 + 0.5 * P.numpy()[0]
            R = 0.5 + 0.5 * R.numpy()[0]
            S = 2*(P * R) / (P + R)
        return P, R, S



def get_dataset_movies():
    db = connect_db('nebula_development')

    query = 'FOR doc IN nebula_comet2020_lsmdc_scored_v03 RETURN doc'
    cursor = db.aql.execute(query, ttl=3600)

    movies_list = []
    for cnt, movie_data in enumerate(cursor):
        movies_list.append(movie_data)

    return movies_list

def comparison_example():
    """
    The function takes ground comparse Eliezer's ground truth with our results
    :return:
    """

    csv_filename = os.path.join(pathlib.Path(__file__).parent.parent.resolve(), 'utils',
                                'vlm-verification-groundtruth.csv')
    # Filter non unicode characterrs
    gt_data = pd.read_csv(csv_filename, encoding='unicode_escape')

    movies_list = get_dataset_movies()
    comp_text = CompareText()

    clipcap_arr = []
    places_arr = []
    actions_arr = []
    for movie in movies_list:
        print(movie)
        clipcap_sent = movie['base'][0]

        movie_id = movie['movie_id']
        s_e = movie['scene_element']
        gt_list = gt_data.index[(gt_data['movie_id'] == movie_id) & (gt_data['stage'] == s_e)].tolist()
        # If no GT continue
        if len(gt_list) == 0:
            continue
        # Remove non unicode characters
        gt_sent = gt_data.iloc[gt_list[0]]['Text'].encode('ascii', 'ignore').decode()

        # Get clipcap rersults
        P, R, F1 = comp_text.compare_sentences(test_sent=clipcap_sent, gt_sent=gt_sent, metric='bert')
        clipcap_arr.append((P, R))

        # Go over all places - choose the best precision
        max_p = 0
        for place in movie['places']:
            P, R, F1 = comp_text.compare_sentences(test_sent=place, gt_sent=gt_sent, metric='bleu')
            if P > max_p:
                max_p = P
        places_arr.append(max_p)

        # Go over all actions - choose the best precision
        max_p = 0
        for action in movie['actions']:
            P, R, F1 = comp_text.compare_sentences(test_sent=action, gt_sent=gt_sent, metric='bleu')
            if P > max_p:
                max_p = P
        actions_arr.append(max_p)

        print('Done computing, save results')

def process_ground_truth_csv():
    csv_filename = os.path.join(pathlib.Path(__file__).parent.parent.resolve(), 'utils',
                                'vlm-verification-groundtruth.csv')
    gt_data = pd.read_csv(csv_filename, encoding='unicode_escape')
    pass
    # with open(csv_filename) as csv_file:
    #     csv_reader = csv.reader(csv_file, delimiter=',', encoding = 'unicode_escape')
    #     for row in csv_reader:
    #         print(row)

    # option 2 - go over the movies in the dataset
    db = connect_db('nebula_development')

    query = 'FOR doc IN nebula_comet2020_lsmdc_scored_v03 RETURN doc'
    cursor = db.aql.execute(query, ttl=3600)

    movies_list = []
    for cnt, movie_data in enumerate(cursor):
        movies_list.append(movie_data)

    # comp_text = CompareText()
    # bertscore_metric = load_metric('bertscore')
    scorer = BERTScorer(lang="en", rescale_with_baseline=True)
    comp_text = CompareText()
    print('Done with movies')
    precision_array = []
    recall_array = []
    f1_array = []
    for movie in movies_list:
        print(movie)
        movie_id = movie['movie_id']
        s_e = movie['scene_element']
        clipcap_sent = movie['base'][0]
        gt_list = gt_data.index[(gt_data['movie_id'] == movie_id) & (gt_data['stage'] == s_e)].tolist()
        if len(gt_list) == 0:
            continue
        gt_sent = gt_data.iloc[gt_list[0]]['Text']
        gt_sent = gt_sent.encode('ascii', 'ignore').decode()

        # Go over all places
        max_p = 0
        for place in movie['actions']:
            place_words = [place.split(' ')]

            bleu_scores = comp_text.compare_sentences(test_sent=place, gt_sent=gt_sent, metric='bleu')
            bert_scores = comp_text.compare_sentences(test_sent=place, gt_sent=gt_sent, metric='bert')
            meteor_scores = comp_text.compare_sentences(test_sent=gt_sent, gt_sent=place)
            meteor_scores = comp_text.compare_sentences(test_sent=[gt_sent], gt_sent=[place])

            # if meteor_scores['rouge1'][0].precision > max_p:
            if meteor_scores['meteor'] > max_p:
                max_p = meteor_scores['meteor']
                # max_p = meteor_scores['rouge1'][0].precision

            # bleu_scores = comp_text.compare_sentences(test_sent=[place.split(' ')], gt_sent=[[gt_sent.split(' ')]],
            #                                           metric='bleu')
            # if bleu_scores['precisions'][0] > max_p:
            #     max_p = bleu_scores['precisions'][0]
            #
            # new_place = []
            # for word in place_words[0]:
            #     if word[-1] == 's':
            #         word = word[:-1]
            #     new_place.append(word)
            #
            # bleu_scores = comp_text.compare_sentences(test_sent=[new_place], gt_sent=[[gt_sent.split(' ')]],
            #                                           metric='bleu')
            # if bleu_scores['precisions'][0] > max_p:
            #     max_p = bleu_scores['precisions'][0]

        # P, R, F1 = scorer.score([clipcap_sent], [gt_sent])
        # bert_scores = bertscore_metric.compute(predictions=[clipcap_sent], references=[gt_sent], lang="en")
        # P = 0.5 + 0.5 * P.numpy()[0]
        # R = 0.5 + 0.5 * R.numpy()[0]
        # F1 = 2*(P * R) / (P + R)
        # precision_array.append(P)
        # recall_array.append(R)
        # f1_array.append(F1)

        precision_array.append(max_p)


    print(precision_array)
    print(recall_array)
    print(f1_array)
    print('Done')

if __name__ == '__main__':
    print('Start processing')
    # process ground truth CSV
    comparison_example()
    process_ground_truth_csv()


    metrics_list = list_metrics()
    print(metrics_list)

    clipcap = ['A woman in a white dress standing next to a man', 'a woman stands in a doorway']
    gt_sent = ['a young surprised blond woman talks to two men near a door and yellow walls',
               'a young woman talks to men in a doorway']

    comp_text = CompareText()
    # meteor assumes that the input is a list of sentences
    # meteor_scores = comp_text.compare_sentences(test_sent=clipcap, gt_sent=gt_sent)
    meteor_scores = comp_text.compare_sentences(test_sent=[clipcap[0]], gt_sent=[gt_sent[0]])
    # precision_scores = comp_text.compare_sentences(test_sent=clipcap, gt_sent=gt_sent, metric='precision')
    # recall_scores = comp_text.compare_sentences(test_sent=clipcap, gt_sent=gt_sent, metric='recall')
    # blue assumes that the input is a list of list of words
    bleu_scores = comp_text.compare_sentences(test_sent=[[clipcap[0].split(' ')]], gt_sent=[[gt_sent[1].split(' ')]],
                                              metric='bleu')
    # bleu_scores['precisions'][0]
    bertscore_metric = load_metric('bertscore')
    print(bertscore_metric)

    bert_scores = bertscore_metric.compute(predictions=[clipcap[0]], references=[gt_sent[0]], lang="en")
    db = connect_db('nebula_development')

    query = 'FOR doc IN nebula_comet2020_lsmdc_scored_v03 RETURN doc'
    cursor = db.aql.execute(query, ttl=3600)

    movies_list = []
    for cnt, movie_data in enumerate(cursor):
        movies_list.append(movie_data)

    print('Done with movies')
