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
            if type(test_sent) == list and type(gt_sent) == list:
                P, R, S = self.curr_metric.score(test_sent, gt_sent)
                P = 0.5 + 0.5 * P.numpy()
                R = 0.5 + 0.5 * R.numpy()
                S = 2 * (P * R) / (P + R)
            else:
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


if __name__ == '__main__':
    print('Start processing')
    # process ground truth CSV
    comparison_example()

    metrics_list = list_metrics()
    print(metrics_list)

    print('Done with movies')
