import os
import json
import gradio as gr
import importlib_resources
from collections import Counter

from sdg.pipeline.box_1 import K1_model
from sdg.pipeline.box_2 import KT_matcher
from sdg.pipeline.box_3 import MyGuidedLDA
from sdg.pipeline.box_4 import K4_model
from sdg.pipeline.box_5 import MyBertTopic


guided_thres = float(os.getenv("GUIDED_THRES", "0.4"))
BERT_thres = float(os.getenv("BERT_THRES", "0.7"))
BERT_thres_old = float(os.getenv("BERT_THRES_OLD", "0.95"))
BERT_ATT_thres_old = float(os.getenv("BERT_ATT_THRES_OLD", "0.98"))
BERTOPIC_score_thres = float(os.getenv("BERTOPIC_SCORE_THRES", "0.14"))
BERTOPIC_count_thres = int(os.getenv("BERTOPIC_COUNT_THRES", "1"))

BASE_PATH = importlib_resources.files(__package__.split(".")[0])


def load_models():
    global k1_1, k1_2, k1_3, kt_match, glda, k4, bertopic
    print(40 * '=')
    print('LOADING bert models')
    k1_1 = K1_model(model_name="distilbert-base-uncased", hidden=100, resume_from=BASE_PATH.joinpath('model_checkpoints/distilbert-base-uncased_100_5e-05_29_84_85.pth.tar'))
    k1_2 = K1_model(model_name="distilbert-base-uncased", hidden=50, resume_from=BASE_PATH.joinpath('model_checkpoints/distilbert-base-uncased_50_5e-05_23_83_84.pth.tar'))
    k1_3 = K1_model(model_name="bert-base-uncased", hidden=100, resume_from=BASE_PATH.joinpath('model_checkpoints/bert-base-uncased_100_5e-05_16_84_84.pth.tar'))

    print(40 * '=')
    print('LOADING KT MATCHING')
    kt_match    = KT_matcher(kt_fpath = BASE_PATH.joinpath('data/sdg_vocabulary.xlsx'))

    print(40 * '=')
    print('LOADING GUIDED LDA')
    glda        = MyGuidedLDA(
        kt_fpath            = BASE_PATH.joinpath('data/sdg_vocabulary.xlsx'),
        guided_tm_path      = BASE_PATH.joinpath('model_checkpoints/guidedlda_model.pickle'),
        guided_tm_cv_path   = BASE_PATH.joinpath('model_checkpoints/guidedlda_countVectorizer.pickle')
    )

    print(40 * '=')
    print('LOADING More bert models')
    k4 = K4_model(
        model_name      = "distilbert-base-uncased",
        resume_from_1   = BASE_PATH.joinpath('model_checkpoints/distilbert-base-uncased_3_87_88.pth.tar'),
        resume_from_2   = BASE_PATH.joinpath('model_checkpoints/distilbert-base-uncased_4_78_80.pth.tar')
    )

    print(40 * '=')
    print('LOADING Bertopic')
    bertopic = MyBertTopic(bert_topic_path = BASE_PATH.joinpath('model_checkpoints/bert_topic_model_sdgs_no_num_of_topics_noemb'))

    print(40 * '=')
    print('DONE LOADING. GO USE IT!')

load_models()

# Define the functions to handle the inputs and outputs
def analyze_text(snippet, progress=gr.Progress(track_tqdm=True)):
    results = {}
    try:
        text_to_check = snippet
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
        results = {
            "final_sdg_categories": dict(Counter(final_sdg_categories)),
            "from_keywords_strict": [kt for kt in kt_sdg_res if kt[-1]>0],
            "from_keywords_lenient": [kt for kt in kt_sdg_res if kt[-1]<=0],
            "from_guided_lda": guided_sdg_res,
            "from_bertopic": bertopic_sdg_res[2],
            "cl_distilbert_100_v2": r1,
            "cl_distilbert_50_v2": r2,
            "cl_bert_100_v2": r3,
            "cl_distilbert_v1": bert_results,
            "cl_distilbert_att_v1": bert_results_att
        }
    except Exception as e:
        results = {'error': str(e)}
    return json.dumps(results)


# Define the interface for the first tab (Text Analysis)
with gr.Blocks() as text_analysis:
    gr.Markdown("### Sustainable Development Goal (SDG) Classifier")
    text_input = gr.Textbox(label="Snippet")
    process_text_button = gr.Button("Process")
    text_output = gr.JSON(label="Output")

    process_text_button.click(analyze_text, inputs=[text_input], outputs=[text_output])

# Combine the tabs into one interface
with gr.Blocks() as demo:
    gr.TabbedInterface([text_analysis], ["Text Mode"])

# Launch the interface
demo.queue().launch()
