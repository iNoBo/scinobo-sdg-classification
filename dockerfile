
FROM ubuntu:18.04

RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y python3.6 python3.6-dev python3-pip

ENV LANG=en_US.UTF-8 \
  LANGUAGE=en_US:en \
  LC_ALL=en_US.UTF-8

RUN mkdir /input_files/
RUN mkdir /output_files/

WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt

ENTRYPOINT ["./berttopic_api/bin/python"]

CMD ["batch_classify.py", "--delimeter=|~|", "--data_path=resources/data/test_input.txt", "--out_path=resources/data/test_output.txt" ]


