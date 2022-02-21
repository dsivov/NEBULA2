from nebula_api.milvus_api import connect_db
from datasets import list_metrics, load_metric


class CompareText:
    def __init__(self):
        self.curr_metric = None

    def compare_sentences(self, test_sent, gt_sent, metric='meteor'):
        """
        :param test_sent: test sentence
        :param gt_sent: ground truth sentence
        :return:
        """
        if self.curr_metric is None or self.curr_metric.name != metric:
            self.curr_metric = load_metric(metric)
        scores = self.curr_metric.compute(predictions=test_sent, references=gt_sent)
        return scores


if __name__ == '__main__':
    print('Start processing')

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
