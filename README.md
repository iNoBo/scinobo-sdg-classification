# scinobo-sdg-classification
A classifier for scientific literature for the UN Sustainable Development Goals (SDG) - developed by SciNoBo-Athena Research Center

## Create Docker image

```
docker build --build-arg HF_TOKEN=<AUTHORIZED ORG HF TOKEN> --tag scinobo_sdg -f ./Dockerfile .
```

## Run server

Run sdg-classification server

```
docker run --gpus all --rm -p 8000:8000 scinobo_sdg uvicorn sdg.server.api:app --host 0.0.0.0 --port 8000
``` 

## Run bulk inference

```
sudo docker run --gpus all --rm \
-v /path/to/input_data:/input_files \
-v /path/to/output_data:/output_files \
scinobo_sdg \
python -m sdg.pipeline.batch_classifier \
--data_path=/input_files/test_input.txt \
--out_path=/output_files/test_output.txt
```