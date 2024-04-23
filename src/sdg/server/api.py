""" 

FastAPI for the SDG classifier. This docstring will be updated.

"""

import logging
import traceback, argparse
import uvicorn
from collections import Counter
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

from sdg.server.logging_setup import setup_root_logger

from pprint import pprint
from sdg.pipeline.box_1 import K1_model
from sdg.pipeline.box_2 import KT_matcher
from sdg.pipeline.box_3 import MyGuidedLDA
from sdg.pipeline.box_4 import K4_model
from sdg.pipeline.box_5 import MyBertTopic

# init the logger
setup_root_logger()
LOGGER = logging.getLogger(__name__)
LOGGER.info("SDG API initialized")

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

################################################################################################################

# declare classes for input-output and error responses

# Pydantic model for the request body
class ClassifierRequest(BaseModel):
    text: str

# Pydantic model for the classification result
class CategoryResult(BaseModel):
    final_sdg_categories: List[str]
    from_keywords_strict: List[Any]
    from_keywords_lenient: List[Any]
    from_guided_lda: Dict[str, float]
    from_bertopic: List[Any]
    cl_distilbert_100_v2: Dict[str, float]
    cl_distilbert_50_v2: Dict[str, float]
    cl_bert_100_v2: Dict[str, float]
    cl_distilbert_v1: Dict[str, float]
    cl_distilbert_att_v1: Dict[str, float]

# Pydantic model for the overall classification response
class ClassifierResponse(BaseModel):
    request: ClassifierRequest
    results: CategoryResult

# Pydantic model for the reload vocabulary result
class VocabularyReloadResult(BaseModel):
    success: int
    message: str
    
# Pydantic model for the reload vocabulary response
class ReloadVocabularyResponse(BaseModel):
    results: VocabularyReloadResult

# Pydantic model for the error response
class ErrorResponse(BaseModel):
    success: int
    message: str

# the FastAPI app
app = FastAPI()

################################################################################################################

guided_thres            = args.guided_thres         #0.4
BERT_thres              = args.BERT_thres           #0.7
BERT_thres_old          = args.BERT_thres_old       #0.95
BERT_ATT_thres_old      = args.BERT_ATT_thres_old   # 0.98
BERTOPIC_score_thres    = args.BERTOPIC_score_thres
BERTOPIC_count_thres    = args.BERTOPIC_count_thres

################################################################################################################

global k1_1, k1_2, k1_3, kt_match, glda, k4, bertopic

def load_models():
    print(40 * '=')
    print('LOADING bert models')
    k1_1 = K1_model(model_name="distilbert-base-uncased", hidden=100, resume_from='./model_checkpoints/distilbert-base-uncased_100_5e-05_29_84_85.pth.tar')
    k1_2 = K1_model(model_name="distilbert-base-uncased", hidden=50, resume_from='./model_checkpoints/distilbert-base-uncased_50_5e-05_23_83_84.pth.tar')
    k1_3 = K1_model(model_name="bert-base-uncased", hidden=100, resume_from='./model_checkpoints/bert-base-uncased_100_5e-05_16_84_84.pth.tar')

    print(40 * '=')
    print('LOADING KT MATCHING')
    kt_match    = KT_matcher(kt_fpath = './data/sdg_vocabulary.xlsx')

    print(40 * '=')
    print('LOADING GUIDED LDA')
    glda        = MyGuidedLDA(
        kt_fpath            = './data/sdg_vocabulary.xlsx',
        guided_tm_path      = './model_checkpoints/guidedlda_model.pickle',
        guided_tm_cv_path   = './model_checkpoints/guidedlda_countVectorizer.pickle'
    )

    print(40 * '=')
    print('LOADING More bert models')
    k4 = K4_model(
        model_name      = "distilbert-base-uncased",
        resume_from_1   = './model_checkpoints/distilbert-base-uncased_3_87_88.pth.tar',
        resume_from_2   = './model_checkpoints/distilbert-base-uncased_4_78_80.pth.tar'
    )

    print(40 * '=')
    print('LOADING Bertopic')
    bertopic = MyBertTopic(bert_topic_path = './model_checkpoints/bert_topic_model_sdgs_no_num_of_topics')

    print(40 * '=')
    print('DONE LOADING. GO USE IT!')

load_models()

# handle CORS -- at a later stage we can restrict the origins
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# create a middleware that logs the requests -- this function logs everything. It might not be needed.
@app.middleware("http")
def log_requests(request, call_next):
    LOGGER.info(f"Request: {request.method} {request.url}")
    response = call_next(request)
    return response

# endpoint for classifying SDG categories
@app.post("/sdg_classifier", response_model=ClassifierResponse, responses={400: {"model": ErrorResponse}})
def sdg_classifier(request: ClassifierRequest):
    try:
        LOGGER.debug(f"JSON received...")
        LOGGER.debug(request.json)
        if request.json:
            request_data = request.json
            pprint(request_data)
            ###################################################################################################
            final_sdg_categories = []
            ######################################
            text_to_check = request_data['text']
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
            results = CategoryResult(
                final_sdg_categories=Counter(final_sdg_categories),
                from_keywords_strict=[kt for kt in kt_sdg_res if kt[-1]>0],
                from_keywords_lenient=[kt for kt in kt_sdg_res if kt[-1]<=0],
                from_guided_lda=guided_sdg_res,
                from_bertopic=bertopic_sdg_res[2],
                cl_distilbert_100_v2=r1,
                cl_distilbert_50_v2=r2,
                cl_bert_100_v2=r3,
                cl_distilbert_v1=bert_results,
                cl_distilbert_att_v1=bert_results_att
            )
            ###################################################################################################
            ret = ClassifierResponse(request=request, results=results)
        else:
            ret = {
                'success': 0,
                'message': 'request should be json formated'
            }
            app.logger.debug(ret)
        return ret
    except Exception as e:
        LOGGER.error(f"Error: {e}")
        raise HTTPException(status_code=400, detail={"success": 0, "message": f"{str(e)}\n{traceback.format_exc()}"})
    

# endpoint for reloading vocabulary
@app.post("/reload_vocabulary", response_model=ReloadVocabularyResponse, responses={500: {"model": ErrorResponse}})
def reload_vocabulary():
    try:
        LOGGER.debug(f"RELOADING SDG KEYPHRASE DICTIONARY...")
        kt_match.reload_vocab()
        results = VocabularyReloadResult(
            success=1,
            message='Finished reloading dictionary.'
        )
        ret = ReloadVocabularyResponse(results=results)
        LOGGER.debug(ret)
        return ret
    except Exception as e:
        LOGGER.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail={"success": 0, "message": f"{str(e)}\n{traceback.format_exc()}"})
    
if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)