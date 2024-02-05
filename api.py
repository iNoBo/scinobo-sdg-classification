#!/usr/bin/env python
# -*- coding: utf-8 -*-

import traceback, argparse
from waitress import serve
from collections import Counter
from flask import Flask
from flask import request
from flask import redirect, url_for, jsonify
from pprint import pprint
from box_1 import K1_model
from box_2 import KT_matcher
from box_3 import MyGuidedLDA
from box_4 import K4_model
from box_5 import MyBertTopic

import logging
logging.basicConfig(filename='sdg_api.log', level=logging.INFO, format="%(asctime)s;%(levelname)s;%(message)s")

################################################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("--host",                   type=str,   default="0.0.0.0",      help="Host for flask api.",     required=False)
parser.add_argument("--port",                   type=int,   default=29929,          help="Port for flask api.",     required=False)
parser.add_argument("--log_path",               type=str,   default='sdg_api_models.log',  help="The path for the log file.", required=False)
parser.add_argument("--guided_thres",           type=float, default=0.4,            help="",                        required=False)
parser.add_argument("--BERT_thres",             type=float, default=0.7,            help="",                        required=False)
parser.add_argument("--BERT_thres_old",         type=float, default=0.95,           help="",                        required=False)
parser.add_argument("--BERT_ATT_thres_old",     type=float, default=0.98,           help="",                        required=False)
parser.add_argument("--BERTOPIC_score_thres",   type=float, default=0.14,           help="",                        required=False)
parser.add_argument("--BERTOPIC_count_thres",   type=int,   default=1,              help="",                        required=False)
args                = parser.parse_args()
logging.basicConfig(filename=args.log_path, level=logging.INFO, format="%(asctime)s;%(levelname)s;%(message)s")

################################################################################################################

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

################################################################################################################

guided_thres            = args.guided_thres         #0.4
BERT_thres              = args.BERT_thres           #0.7
BERT_thres_old          = args.BERT_thres_old       #0.95
BERT_ATT_thres_old      = args.BERT_ATT_thres_old   # 0.98
BERTOPIC_score_thres    = args.BERTOPIC_score_thres
BERTOPIC_count_thres    = args.BERTOPIC_count_thres

################################################################################################################

print(40 * '=')
print('LOADING bert models')
k1_1 = K1_model(model_name="distilbert-base-uncased", hidden=100, resume_from='./models/distilbert-base-uncased_100_5e-05_29_84_85.pth.tar')
k1_2 = K1_model(model_name="distilbert-base-uncased", hidden=50, resume_from='./models/distilbert-base-uncased_50_5e-05_23_83_84.pth.tar')
k1_3 = K1_model(model_name="bert-base-uncased", hidden=100, resume_from='./models/bert-base-uncased_100_5e-05_16_84_84.pth.tar')

print(40 * '=')
print('LOADING KT MATCHING')
kt_match    = KT_matcher(kt_fpath = './models/sdg_vocabulary.xlsx')

print(40 * '=')
print('LOADING GUIDED LDA')
glda        = MyGuidedLDA(
    kt_fpath            = './models/sdg_vocabulary.xlsx',
    guided_tm_path      = './models/guidedlda_model.pickle',
    guided_tm_cv_path   = './models/guidedlda_countVectorizer.pickle'
)

print(40 * '=')
print('LOADING More bert models')
k4 = K4_model(
    model_name      = "distilbert-base-uncased",
    resume_from_1   = './models/distilbert-base-uncased_3_87_88.pth.tar',
    resume_from_2   = './models/distilbert-base-uncased_4_78_80.pth.tar'
)

print(40 * '=')
print('LOADING Bertopic')
bertopic = MyBertTopic(bert_topic_path = '/media/disk1/dpappas_data/BERTopic/bert_topic_model_sdgs_no_num_of_topics')

print(40 * '=')
print('DONE LOADING. GO USE IT!')

################################################################################################################

@app.route('/sdg_classifier', methods=['GET','POST'])
def sdg_classifier():
    try:
        app.logger.debug("JSON received...")
        app.logger.debug(request.json)
        if request.json:
            mydata = request.json
            pprint(mydata)
            ###################################################################################################
            final_sdg_categories = []
            ######################################
            text_to_check = mydata['text']
            ######################################
            kt_sdg_res          = kt_match.emit_for_abstracts([text_to_check])[0][1]
            # kt_sdg_res_counter  = Counter([t[0] for t in kt_sdg_res])
            ######################################
            r1 = k1_1.emit_for_abstracts([text_to_check])[0][1]
            r2 = k1_2.emit_for_abstracts([text_to_check])[0][1]
            r3 = k1_3.emit_for_abstracts([text_to_check])[0][1]
            ###################################################################################################
            bert_results, bert_results_att = k4.emit_for_abstracts([text_to_check])
            bert_results, bert_results_att = bert_results[0][1], bert_results_att[0][1]
            ###################################################################################################
            guided_sdg_res      = glda.emit_for_abstracts([text_to_check])[0][1]
            ###################################################################################################
            bertopic_sdg_res    = bertopic.emit_for_abstracts(
                abstracts           = [text_to_check],
                threshold_on_score  = BERTOPIC_score_thres,
                threshold_on_count  = BERTOPIC_count_thres
            )[0]
            ###################################################################################################
            # final_sdg_categories += list(kt_sdg_res_counter.keys())
            # final_sdg_categories += list([k for k in kt_sdg_res_counter if kt_sdg_res_counter[k]>=2])
            final_sdg_categories  = [t[0] for t in kt_sdg_res]
            final_sdg_categories += [k for k, v in guided_sdg_res.items() if v > guided_thres]
            final_sdg_categories += bertopic_sdg_res[1]
            final_sdg_categories += [k for k, v in r1.items() if v >= BERT_thres]
            final_sdg_categories += [k for k, v in r2.items() if v >= BERT_thres]
            final_sdg_categories += [k for k, v in r3.items() if v >= BERT_thres]
            final_sdg_categories += [k for k, v in bert_results.items() if v > BERT_thres_old]
            final_sdg_categories += [k for k, v in bert_results_att.items() if v > BERT_ATT_thres_old]
            ###################################################################################################
            pprint(list(set(final_sdg_categories)))
            ###################################################################################################
            ret = {
                'request'               : mydata,
                'results'               : {
                    'final_sdg_categories'  : Counter(final_sdg_categories),
                    'from_keywords_strict'  : [kt for kt in kt_sdg_res if kt[-1]>0],
                    'from_keywords_lenient' : [kt for kt in kt_sdg_res if kt[-1]<=0],
                    'from_guided_lda'       : guided_sdg_res,
                    'from_bertopic'         : bertopic_sdg_res[2],
                    'cl_distilbert_100_v2'  : r1,
                    'cl_distilbert_50_v2'   : r2,
                    'cl_bert_100_v2'        : r3,
                    'cl_distilbert_v1'     : bert_results,
                    'cl_distilbert_att_v1' : bert_results_att
                }
            }
            ###################################################################################################
            return jsonify(ret)
        else:
            ret = {
                'success': 0,
                'message': 'request should be json formated'
            }
            app.logger.debug(ret)
            return jsonify(ret)
    except Exception as e:
        app.logger.debug(str(e))
        traceback.print_exc()
        ret = {'success': 0, 'message': str(e) + '\n' + traceback.format_exc()}
        app.logger.debug(ret)
        return jsonify(ret)

@app.route('/reload_vocabulary', methods=['GET','POST'])
def reload_vocabulary():
    try:
        app.logger.debug("RELOADING SDG KEYPHRASE DICTIONARY...")
        kt_match.reload_vocab()
        ret = {'success': 1, 'message': 'Finished reloading dictionary.'}
        app.logger.debug(ret)
        return jsonify(ret)
    except Exception as e:
        app.logger.debug(str(e))
        traceback.print_exc()
        ret = {'success': 0, 'message': str(e) + '\n' + traceback.format_exc()}
        app.logger.debug(ret)
        return jsonify(ret)

################################################################################################################

if __name__ == '__main__':
    serve(app, host=args.host, port=int(args.port))


# ./berttopic_api/bin/python api.py

