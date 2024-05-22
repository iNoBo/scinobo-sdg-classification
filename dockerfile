FROM ubuntu:18.04

RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y python3.8 python3.8-dev python3-pip
RUN apt-get install -y vim

RUN ln -sfn /usr/bin/python3.8  /usr/bin/python3
RUN ln -sfn /usr/bin/python3    /usr/bin/python
RUN ln -sfn /usr/bin/python3    /usr/bin/python

ENV LANG=C.UTF-8 LANGUAGE=C.UTF-8 LC_ALL=C.UTF-8

RUN mkdir /input_files/
RUN mkdir /output_files/

WORKDIR /app
COPY . /app

RUN mkdir /local_models
RUN mkdir /global_models

RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

RUN cp -a /usr/local/lib/python3.8/dist-packages/lda/ /usr/local/lib/python3.8/dist-packages/guidedlda/
RUN cp /app/GuidedLDA_WorkAround/*.py /usr/local/lib/python3.8/dist-packages/guidedlda/
RUN cp /app/GuidedLDA_WorkAround/*.py /usr/local/lib/python3.8/dist-packages/lda/

# Make dirs
RUN mkdir -p /app/src/sdg/model_checkpoints/

RUN apt-get install -y wget

# Get the huggingface token from the build args
ARG HF_TOKEN

# Download model checkpoint
RUN wget https://huggingface.co/iNoBo/scinobo-sdg-classification-bert-topic/resolve/main/bert_topic_model_sdgs_no_num_of_topics?download=true -O /app/src/sdg/model_checkpoints/bert_topic_model_sdgs_no_num_of_topics

COPY . /app

# Change working directory
WORKDIR /app/src

# Envs for uvicorn api
ENV HOST="0.0.0.0"
ENV LOG_PATH="sdg_api_models.log"
ENV GUIDED_THRES="0.4"
ENV BERT_THRES="0.7"
ENV BERT_THRES_OLD="0.95"
ENV BERT_ATT_THRES_OLD="0.98"
ENV BERTOPIC_SCORE_THRES="0.14"
ENV BERTOPIC_COUNT_THRES="1"

EXPOSE 8000

# Run a shell
CMD ["bash"]
