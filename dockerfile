FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

RUN mkdir /input_files/
RUN mkdir /output_files/

WORKDIR /app

# Install wget
RUN apt-get update && apt-get install -y wget build-essential

# Copy only the requirements file, to cache the installation of dependencies
COPY requirements.txt /app/requirements.txt

# COPY DESCRIPTIONS
# install dependencies
RUN pip install -r requirements.txt

# Expose the port the app runs on
EXPOSE 8000

# Set a default port
ENV PORT=8000

# Get the huggingface token from the build args
ARG HF_TOKEN

# Make dirs
RUN mkdir -p /app/src/sdg/model_checkpoints/

# Download model checkpoint
RUN wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/iNoBo/scinobo-sdg-classification-bert-topic/resolve/main/bert_topic_model_sdgs_no_num_of_topics?download=true -O /app/src/sdg/model_checkpoints/bert_topic_model_sdgs_no_num_of_topics

# Copy the rest of your application
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

# Run a shell
CMD ["bash"]

#CMD ["batch_classify.py", "--delimeter=|~|", "--data_path=resources/data/test_input.txt", "--out_path=resources/data/test_output.txt" ]


