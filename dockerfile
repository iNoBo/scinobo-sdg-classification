FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

RUN mkdir /input_files/
RUN mkdir /output_files/

WORKDIR /app

# Copy only the requirements file, to cache the installation of dependencies
COPY requirements.txt /app/requirements.txt

# COPY DESCRIPTIONS
# install dependencies
RUN pip install -r requirements.txt

# Expose the port the app runs on
EXPOSE 8000

# Set a default port
ENV PORT=8000

# Copy the rest of your application
COPY . /app

ENTRYPOINT ["./berttopic_api/bin/python"]

# Change working directory
WORKDIR /app/src

# Run a shell
CMD ["bash"]

#CMD ["batch_classify.py", "--delimeter=|~|", "--data_path=resources/data/test_input.txt", "--out_path=resources/data/test_output.txt" ]


