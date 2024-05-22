

# virtualenv --python=python3.6 transformers
# source /home/dpappas/venvs/transformers/bin/activate
# pip install transformers[sentencepiece]
# pip install torch==1.10.2

# source /media/dpappas/dpappas_data/sdg_classifier_api/sdg_api_venv/bin/activate
# /media/dpappas/dpappas_data/sdg_classifier_api/sdg_api_venv/bin/python

import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from pprint import pprint

class Ontop_Modeler(nn.Module):
    def __init__(self, catname_to_catid):
        super(Ontop_Modeler, self).__init__()
        self.att_layer              = nn.Linear(768, 1, bias=True)
        self.linear1                = nn.Linear(768, len(catname_to_catid), bias=True)
        self.loss                   = nn.CrossEntropyLoss()
        self.sfmax                  = nn.Softmax(dim=1)
    def emit(self, input_xs, maska):
        attention       = self.att_layer(input_xs)
        attention       = self.masked_softmax(attention, maska.unsqueeze(-1), dim=1)
        ##########################################################################################
        attended_vec    = torch.bmm(attention.transpose(-1,-2), input_xs).squeeze(1)
        output          = self.linear1(attended_vec)
        output          = self.sfmax(output)
        return output, attention
    def masked_softmax(self, vec, mask, dim=1):
        masked_vec = vec * mask.float()
        max_vec = torch.max(masked_vec, dim=dim, keepdim=True)[0]
        exps = torch.exp(masked_vec - max_vec)
        masked_exps = exps * mask.float()
        masked_sums = masked_exps.sum(dim, keepdim=True)
        zeros = (masked_sums == 0)
        masked_sums += zeros.float()
        return masked_exps / masked_sums
    def forward(self, input_xs, maska, input_target):
        attention       = self.att_layer(input_xs)
        attention       = self.masked_softmax(attention, maska.unsqueeze(-1), dim=1)
        ##########################################################################################
        attended_vec    = torch.bmm(attention.transpose(-1,-2), input_xs).squeeze(1)
        output          = self.linear1(attended_vec)
        loss            = self.loss(output, input_target)
        output          = self.sfmax(output)
        return output, loss

class Ontop_Modeler_2(nn.Module):
    def __init__(self, catname_to_catid):
        super(Ontop_Modeler_2, self).__init__()
        self.linear1                = nn.Linear(768, len(catname_to_catid), bias=True)
        # self.linear1                = nn.Linear(768, 100, bias=True)
        # self.linear2                = nn.Linear(100, len(catname_to_catid), bias=True)
        self.loss                   = nn.CrossEntropyLoss()
        self.sfmax                  = nn.Softmax(dim=1)
        # self.tanh                   = nn.Tanh()
    def emit(self, input_xs):
        y     = self.linear1(input_xs)
        # y     = self.tanh(y)
        # y     = self.linear2(y)
        y     = self.sfmax(y)
        return y
    def forward(self, input_xs, input_target):
        y     = self.linear1(input_xs)
        # y     = self.tanh(y)
        # y     = self.linear2(y)
        loss  = self.loss(y, input_target)
        y     = self.sfmax(y)
        return y, loss

class K4_model:
    def __init__(
        self,
        model_name      = "distilbert-base-uncased",
        resume_from_1   = None,
        resume_from_2   = None
    ):
        self.resume_from_1      = resume_from_1
        self.resume_from_2      = resume_from_2
        self.model_name         = model_name
        self.use_cuda           = torch.cuda.is_available()
        self.device             = torch.device("cuda") if (self.use_cuda) else torch.device("cpu")
        self.catid_to_catname   = {
            0: '8. Economic growth',
            1: '6. Clean water',
            2: '13. Climate action',
            3: '14. Life underwater',
            4: '16. Peace & justice',
            5: '1. No poverty',
            6: '10. No inequality',
            7: '15. Life on land',
            8: '11. Sustainability',
            9: '3. Good health'
        }
        self.catname_to_catid   = dict((v, k) for k, v in self.catid_to_catname.items())
        #########################################################################################
        (
            self.my_model, self.my_model_2, self.bert_tokenizer, self.bert_model, self.catid_to_catname
        ) = self.load_bert_models()
    def load_bert_models(self):
        #######################################################################################
        bert_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        pprint(bert_tokenizer.special_tokens_map)
        bert_model = AutoModel.from_pretrained(self.model_name).to(self.device)
        gb = bert_model.eval()
        #######################################################################################
        my_model = Ontop_Modeler(self.catname_to_catid).to(self.device)
        self.load_model_from_checkpoint(my_model, self.resume_from_1)
        my_model.eval()
        #######################################################################################
        my_model_2 = Ontop_Modeler_2(self.catname_to_catid).to(self.device)
        self.load_model_from_checkpoint(my_model_2, self.resume_from_2)
        my_model_2.eval()
        #######################################################################################
        return my_model, my_model_2, bert_tokenizer, bert_model, self.catid_to_catname
    def embed_abstracts(self, abstracts):
        use_cuda = False  # torch.cuda.is_available()
        inputs = self.bert_tokenizer(abstracts, padding=True, truncation=True, return_tensors="pt")
        bpe_mask = inputs['attention_mask'].to(self.device)
        with torch.no_grad():
            pooled = self.bert_model(**inputs.to(self.device))[0]
        return pooled, bpe_mask
    def embed_abstracts_2(self, abstracts):
        use_cuda = False  # torch.cuda.is_available()
        bpe_ids = self.bert_tokenizer(abstracts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            pooled = self.bert_model(**bpe_ids.to(self.device))[0][:, 0, :]
        return pooled
    def load_model_from_checkpoint(self, my_model, resume_from):
        global start_epoch, optimizer
        if os.path.isfile(resume_from):
            print("=> loading checkpoint '{}'".format(resume_from))
            checkpoint = torch.load(resume_from, map_location=lambda storage, loc: storage)
            my_model.load_state_dict(checkpoint['state_dict'])
        else:
            print("=> could not find path !!! '{}'".format(resume_from))
    def emit_for_abstracts(self, abstracts):
        r1 = []
        with torch.no_grad():
            abs_vecs, mask  = self.embed_abstracts(abstracts)
            bert_out, mask_ = self.my_model.emit(abs_vecs, mask)
            bert_out        = bert_out.cpu().data.numpy().tolist()
        ######################################
        r2 = []
        with torch.no_grad():
            abs_vec         = self.embed_abstracts_2(abstracts)
            bert_att_out    = self.my_model_2.emit(abs_vec).cpu().data.numpy().tolist()
        ######################################
        for j in range(len(abstracts)):
            r1.append(
                (
                    abstracts[j],
                    dict((self.catid_to_catname[i], round(bert_out[j][i], 4)) for i in range(len(bert_out[j])))
                )
            )
            r2.append(
                (
                    abstracts[j],
                    dict((self.catid_to_catname[i], round(bert_att_out[j][i], 4)) for i in range(len(bert_att_out[j])))
                )
            )
        ######################################
        return r1, r2

if __name__ == '__main__':
    abstracts = [
        '''
        HIV disproportionately impacts youth, particularly young men who have sex with men (YMSM), a population that includes subgroups of young men who have sex with men only (YMSMO) and young men who have sex with men and women (YMSMW). In 2015, among male youth, 92% of new HIV diagnoses were among YMSM. The reasons why YMSM are disproportionately at risk for HIV acquisition, however, remain incompletely explored. We performed event-level analyses to compare how the frequency of condom use, drug and/or alcohol use at last sex differed among YMSMO and YMSWO (young men who have sex with women only) over a ten-year period from 2005–2015 within the Youth Risk Behavior Survey (YRBS). YMSMO were less likely to use condoms at last sex compared to YMSWO. However, no substance use differences at last sexual encounter were detected. From 2005–2015, reported condom use at last sex significantly declined for both YMSMO and YMSWO, though the decline for YMSMO was more notable. While there were no significant differences in alcohol and substance use at last sex over the same ten-year period for YMSMO, YMSWO experienced a slight but significant decrease in reported alcohol and substance use. These event-level analyses provide evidence that YMSMO, similar to adult MSMO, may engage in riskier sexual behaviors compared to YMSWO, findings which may partially explain the increased burden of HIV in this population. Future work should investigate how different patterns of event-level HIV risk behaviors vary over time among YMSMO, YMSWO, and YMSMW, and are tied to HIV incidence among these groups.
        '''.strip(),
        "french populism and discourses on secularism nilsson pereriknew york bloomsbury  pp  isbn  since the first controversies over hijabs back in  laicitethe particular french version of secularismhas increasingly been used as a legitimizing frame for an exclusionary agenda aimed at re".strip(),
        '''
        comprendre des histoires en cours prparatoire lexemple du rappel de rcit accompagn cette tude propose une analyse qualitative de trois sances de rappel de rcit choisies afin de mieux cerner les gestes professionnels denseignants de cours prparatoire dans le domaine de la comprhension de textes les interactions orales lors de ces rappels de rcit prsentent des caractristiques communes le questionnement de lenseignant facilite la caractrisation des personnages vise  expliciter leurs penses et leurs actions les reformulations aident  coconstruire le rcit lclaircissement du lexique guide les lves grce  un retour systmatique  lnoncsource ces modalits reprsenteraient des gestes didactiques fondamentaux tayant la comprhension notamment pour les lves les plus en difficult this study offers a qualitative analysis of three selected teaching practices in an attempt to identify the professional actions of teachers of reading comprehension the collective moments devoted to retelling seem to present a certain number of common characteristics the questioning techniques of the teacher help with the description of the characters aims to explain their thoughts and their actions the reformulation of the original text and the pupils suggestions help to coconstruct the story the clarification of single words in the text guide the pupils through a systematic return to the source text all these methods seem to constitute fundamental educational actions which underpin comprehension especially for those pupils who find it most difficult
        '''.strip(),
    ]
    k4 = K4_model(
        model_name      = "distilbert-base-uncased",
        resume_from_1   = '/media/dpappas/dpappas_data/sdg_classifier_api/distilbert-base-uncased_3_87_88.pth.tar',
        resume_from_2   = '/media/dpappas/dpappas_data/sdg_classifier_api/distilbert-base-uncased_4_78_80.pth.tar'
    )
    bert_results, bert_results_att = k4.emit_for_abstracts(abstracts)
    print(40*'=')
    for abs, sdg_cats in bert_results:
        print(abs)
        pprint(sdg_cats)
        print(40*'-')
    print(20*'=')
    for abs, sdg_cats in bert_results_att:
        print(abs)
        pprint(sdg_cats)
        print(40*'-')
    ######################################################################################################




