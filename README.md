# scinobo-sdg-classification
A classifier for scientific literature for the UN Sustainable Development Goals (SDG) - developed by SciNoBo-Athena Research Center

## Create Docker image

```
sudo docker build --build-arg HF_TOKEN=<AUTHORIZED ORG HF TOKEN> --tag scinobo_sdg -f ./Dockerfile .
```

## Run image container as a demo

If there is already a container running remove it

```
sudo docker stop sdg_black_box_1
sudo docker rm   sdg_black_box_1
``` 

Then we run an image container 

```
sudo docker container run -d -it --name sdg_black_box_1 -i scinobo_sdg
```

Check whether the container is running

```
sudo docker container ls --all
sudo docker logs sdg_black_box_1
``` 

and Collect the output of the classifier
 
 ```
 sudo docker cp sdg_black_box_1:app/resources/data/test_output.txt ./
```
 

## Run image container in production

You have to mount the input directory and the output directory

```
-v input_directory:directory_inside_docker"
-v output_directory:directory_outside_docker"
``` 

Example:

 ```

sudo docker run \
-v /path/to/input_data:/input_files \
-v /path/to/output_data:/output_files \
-i intelcomp_sdg \
batch_classifier.py \
--distilbert_path=/app/distilbert-base-uncased/snapshots/1c4513b2eedbda136f57676a34eea67aba266e5c/ \
--bert_path=/app/bert-base-uncased/snapshots/0a6aa9128b6194f4f3c4db429b6cb4891cdb421b/ \
--data_path=/input_files/test_input.txt \
--out_path=/output_files/test_output.txt
