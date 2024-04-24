


import argparse, pickle, torch
from pprint import pprint
from sdg.pipeline.box_1 import K1_model
from sdg.pipeline.box_2 import KT_matcher
from sdg.pipeline.box_3 import MyGuidedLDA
from sdg.pipeline.box_4 import K4_model
from sdg.pipeline.box_5 import MyBertTopic
from tqdm import tqdm
from collections import Counter
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
import pyarrow as pa

import importlib_resources

################################################################################################################

import logging
logging.basicConfig(filename='sdg_batch_parquet.log', level=logging.INFO, format="%(asctime)s;%(levelname)s;%(message)s")
BASE_PATH = importlib_resources.files(__package__.split(".")[0])

################################################################################################################

parser  = argparse.ArgumentParser()
parser.add_argument("--data_path",          type=str,   default="./test_input.parquet",   help="",              required=False)
parser.add_argument("--out_path",           type=str,   default="./test_output.parquet",  help="{doi:Counter()}",         required=False)
parser.add_argument("--log_path",           type=str,   default='sdg_batch_parquet_models.log',  help="The path for the log file.", required=False)
parser.add_argument("--guided_thres",       type=float, default=0.4,            help="",                        required=False)
parser.add_argument("--batch_size",         type=int,   default=100,            help="",                        required=False)
parser.add_argument("--BERT_thres",         type=float, default=0.7,            help="",                        required=False)
parser.add_argument("--BERT_thres_old",     type=float, default=0.95,           help="",                        required=False)
parser.add_argument("--BERT_ATT_thres_old", type=float, default=0.98,           help="",                        required=False)
parser.add_argument("--BERTOPIC_score_thres",   type=float, default=0.14,       help="",                        required=False)
parser.add_argument("--BERTOPIC_count_thres",   type=int,   default=1,          help="",                        required=False)
parser.add_argument("--ensemble_agreement",     type=int,   default=3,          help="",                        required=False)
args    = parser.parse_args()
logging.basicConfig(filename=args.log_path, level=logging.INFO, format="%(asctime)s;%(levelname)s;%(message)s")

################################################################################################################

guided_thres            = args.guided_thres         #0.4
BERT_thres              = args.BERT_thres           #0.7
BERT_thres_old          = args.BERT_thres_old       #0.95
BERT_ATT_thres_old      = args.BERT_ATT_thres_old   # 0.98
BERTOPIC_score_thres    = args.BERTOPIC_score_thres
BERTOPIC_count_thres    = args.BERTOPIC_count_thres
data_path               = args.data_path
batch_size              = args.batch_size
out_path                = args.out_path
ensemble_agreement      = args.ensemble_agreement

################################################################################################################

print(40 * '=')
print('LOADING bert models')
k1_1 = K1_model(model_name="distilbert-base-uncased", hidden=100,   resume_from=BASE_PATH.joinpath('model_checkpoints/distilbert-base-uncased_100_5e-05_29_84_85.pth.tar'))
k1_2 = K1_model(model_name="distilbert-base-uncased", hidden=50,    resume_from=BASE_PATH.joinpath('model_checkpoints/distilbert-base-uncased_50_5e-05_23_83_84.pth.tar'))
k1_3 = K1_model(model_name="bert-base-uncased", hidden=100,         resume_from=BASE_PATH.joinpath('model_checkpoints/bert-base-uncased_100_5e-05_16_84_84.pth.tar'))

print(40 * '=')
print('LOADING More bert models')
k4 = K4_model(
    model_name      = "distilbert-base-uncased",
    resume_from_1   = BASE_PATH.joinpath('model_checkpoints/distilbert-base-uncased_3_87_88.pth.tar'),
    resume_from_2   = BASE_PATH.joinpath('model_checkpoints/distilbert-base-uncased_4_78_80.pth.tar')
)

print(40 * '=')
print('LOADING KT MATCHING')
kt_match    = KT_matcher(kt_fpath = BASE_PATH.joinpath('model_checkpoints/sdg_vocabulary.xlsx'))

print(40 * '=')
print('LOADING GUIDED LDA')
glda        = MyGuidedLDA(
    kt_fpath            = BASE_PATH.joinpath('model_checkpoints/sdg_vocabulary.xlsx'),
    guided_tm_path      = BASE_PATH.joinpath('model_checkpoints/guidedlda_model.pickle'),
    guided_tm_cv_path   = BASE_PATH.joinpath('model_checkpoints/guidedlda_countVectorizer.pickle')
)
print(40 * '=')
print('LOADING Bertopic')
if torch.cuda.is_available():
    bertopic = MyBertTopic(bert_topic_path = BASE_PATH.joinpath('model_checkpoints/bert_topic_model_sdgs_no_num_of_topics'))
else:
    bertopic = MyBertTopic(bert_topic_path=BASE_PATH.joinpath('model_checkpoints/bert_topic_model_sdgs_no_num_of_topics_CPU'))

print(40 * '=')
print('DONE LOADING. GO USE IT!')

################################################################################################################

def do_for_one_batch(batch_dois, batch_texts, all_file_results):
    ################################################################################
    kt_sdg_res              = kt_match.emit_for_abstracts(batch_texts)
    r1                      = k1_1.emit_for_abstracts(batch_texts)
    r2                      = k1_2.emit_for_abstracts(batch_texts)
    r3                      = k1_3.emit_for_abstracts(batch_texts)
    bert_results, bert_results_att = k4.emit_for_abstracts(batch_texts)
    guided_sdg_res          = glda.emit_for_abstracts(batch_texts)
    bertopic_sdg_res        = bertopic.emit_for_abstracts(
        abstracts           = batch_texts,
        threshold_on_score  = BERTOPIC_score_thres,
        threshold_on_count  = BERTOPIC_count_thres
    )
    ################################################################################
    for i in range(len(batch_dois)):
        bdoi                = batch_dois[i]
        # each item of kt_sdg_res is (original_text, kt_results)
        ############################################################################
        # kt_results is a list where each element is (sdg, tuple(kt), le_count)
        bkt_sdg_res         = kt_sdg_res[i][1]
        # kt_results is a list where each element is (sdg, tuple(kt), le_count)
        bkt_sdg_res         = [[t[0]]* t[-1] for t in bkt_sdg_res]
        # Here maybe we should change it. We should multiply the SDG_category with the times found. Since these are keywords it makes sense
        bkt_sdg_res         = [x for xs in bkt_sdg_res for x in xs] # flatten
        ############################################################################
        bguided_sdg_res     = guided_sdg_res[i][1]
        bbertopic_sdg_res   = bertopic_sdg_res[i][1]
        br1                 = r1[i][1]
        br2                 = r2[i][1]
        br3                 = r3[i][1]
        bbert_results       = bert_results[i][1]
        bbert_results_att   = bert_results_att[i][1]
        final_sdg_categories = []
        #
        final_sdg_categories += bkt_sdg_res
        # final_sdg_categories += list(Counter([t[0] for t in bkt_sdg_res]).keys())
        final_sdg_categories += [k for k, v in bguided_sdg_res.items() if v > guided_thres]
        final_sdg_categories += bbertopic_sdg_res
        final_sdg_categories += [k for k, v in br1.items() if v >= BERT_thres]
        final_sdg_categories += [k for k, v in br2.items() if v >= BERT_thres]
        final_sdg_categories += [k for k, v in br3.items() if v >= BERT_thres]
        final_sdg_categories += [k for k, v in bbert_results.items() if v > BERT_thres_old]
        final_sdg_categories += [k for k, v in bbert_results_att.items() if v > BERT_ATT_thres_old]
        if bdoi in all_file_results:
            all_file_results[bdoi].update(Counter(final_sdg_categories))
        else:
            all_file_results[bdoi] = Counter(final_sdg_categories)
    ################################################################################
    return all_file_results

if __name__ == '__main__':
    ################################################################################
    table2 = pq.read_table(data_path, columns=['id', 'title', 'abstract'])
    table2 = table2.to_pandas()
    table2["text"] = table2['title'].astype(str) + "\n" + table2["abstract"]
    ################################################################################
    all_file_results    = {}
    fromm               = 0
    for fromm in tqdm(range(0, table2.shape[0], batch_size)):
        batch_dois          = table2[fromm:fromm+batch_size]['id'].to_list()
        batch_texts         = table2[fromm:fromm+batch_size]['text'].to_list()
        all_file_results    = do_for_one_batch(batch_dois, batch_texts, all_file_results)
    ################################################################################
    export_data         = []
    for doi in all_file_results:
        for sdg_cat_, sdg_score_ in all_file_results[doi].items():
            if sdg_score_ >= ensemble_agreement:
                export_data.append([doi, sdg_cat_, sdg_score_])
    ################################################################################
    output_df   = pd.DataFrame(export_data, columns=['id', 'sdg_category', 'score'])
    table       = pa.Table.from_pandas(output_df, preserve_index=False)
    pq.write_table(table, out_path)
    ################################################################################
    print('COMPLETED')
    ################################################################################


'''
import pyarrow.parquet as pq
table2	= pq.read_table('/storage3/dpappas/rarediseases.parquet', columns=['id', 'title', 'abstract'])
table2	= table2.to_pandas()
table2	= table2[:50]
table2	= pa.Table.from_pandas(table2, preserve_index=False)
pq.write_table(table2, '/storage3/dpappas/sdg_docker_feb_24/test_input.parquet')
'''
