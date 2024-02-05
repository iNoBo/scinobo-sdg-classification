

from bertopic import BERTopic
from pprint import pprint
from collections import Counter

class MyBertTopic:
    def __init__(self, bert_topic_path = '/home/dpappas/guidedlda_countVectorizer.pickle'):
        self.bert_topic_path    = bert_topic_path
        self.topic_model        = BERTopic.load(self.bert_topic_path)
        self.topic_to_sdg       = {
            0 : [0], 1 : [16], 2 : [6], 3 : [4], 4 : [0], 5 : [2, 15], 6 : [2], 7 : [0], 8 : [1], 9 : [15], 10 : [3],
            11 : [3, 10], 12 : [7, 11], 13 : [3], 14 : [5], 15 : [11], 16 : [14], 17 : [0], 18 : [3], 19 : [1, 8],
            20 : [2], 21 : [2], 22 : [11, 13], 23 : [0], 24 : [12], 25 : [15], 26 : [5], 27 : [8], 28 : [0], 29 : [13],
            30 : [8], 31 : [13, 15], 32 : [4, 10], 33 : [0], 34 : [9], 35 : [7], 36 : [13, 14], 37 : [13, 14], 38 : [12],
            39 : [3], 40 : [0], 41 : [0], 42 : [8, 11, 12], 43 : [7], 44 : [3], 45 : [11, 12], 46 : [7], 47 : [7],
            48 : [3], 49 : [7], 50 : [0], 51 : [3], 52 : [4], 53 : [7], 54 : [5], 55 : [7], 56 : [5, 10], 57 : [3],
            58 : [13, 14], 59 : [0], 60 : [14], 61 : [3]
        }
        self.catid_to_catname   = {
            1:  '1. No poverty',
            2:  '2. Zero hunger',
            3:  '3. Good health',
            4:  '4. Education',
            5:  '5. Gender equality',
            6:  '6. Clean water',
            7:  '7. Clean energy',
            8:  '8. Economic growth',
            9:  '9. Industry and infrastructure',
            10: '10. No inequality',
            11: '11. Sustainability',
            12: '12. Responsible consumption',
            13: '13. Climate action',
            14: '14. Life underwater',
            15: '15. Life on land',
            16: '16. Peace & justice'
        }
    def emit_for_abstracts(self, abstracts, threshold_on_score=0.14, threshold_on_count=1):
        ret = []
        for i in range(len(abstracts)):
            kept_sdgs = []
            topics_found, topics_score  = self.topic_model.find_topics(abstracts[i])
            for t, s in zip(topics_found, topics_score):
                if s >= threshold_on_score:
                    kept_sdgs.extend(self.topic_to_sdg[t+1])
            les_sdgs = [x[0] for x in Counter(kept_sdgs).items() if x[1]>=threshold_on_count]
            les_sdgs = [self.catid_to_catname[idd] for idd in les_sdgs if idd!=0]
            ret.append(
                (
                    abstracts[i],
                    les_sdgs,
                    dict(
                        (idd, {'score' : scoree, 'sdgs'  : self.topic_to_sdg[idd+1]})
                        for (idd, scoree) in zip(topics_found, topics_score)
                    )
                )
            )
        return ret

if __name__ == '__main__':
    some_text = '''
    Stigma, Discrimination, and Mental Health Outcomes Among Transgender Women With Diagnosed HIV Infection in the United States, 2015-2018.
    OBJECTIVE
    Transgender women with diagnosed HIV experience social and structural factors that could negatively affect their overall health and HIV-related health outcomes. 
    We describe estimates from the Centers for Disease Control and Prevention Medical Monitoring Project (MMP) of sociodemographic characteristics, HIV stigma, discrimination, and mental health outcomes among transgender women with diagnosed HIV.
    '''.strip()
    ######################################################################################################
    k5      = MyBertTopic(bert_topic_path = '/media/dpappas/dpappas_data/BERTopic/bert_topic_model_sdgs_no_num_of_topics')
    res     = k5.emit_for_abstracts(3*[some_text])
    print(40*'=')
    for abs, sdg_cats in res:
        print(abs)
        pprint(sdg_cats)
        print(40*'-')
    ######################################################################################################






