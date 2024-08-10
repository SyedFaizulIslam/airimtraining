FROM python:3.8-slim
COPY ./main.py /deploy/
COPY ./ChatPrompt.py /deploy/
#COPY ./AirimMongoClient.py /deploy/
#COPY ./AzureBlobStorageClient.py /deploy/
#COPY ./CommonMethods.py /deploy/
COPY ./nltk_data /deploy/nltk_data/
COPY ./app /deploy/app/
COPY ./requirements.txt /deploy/




RUN pip install --upgrade pip
RUN apt-get -y clean
WORKDIR /deploy/
RUN pip3 install --default-timeout=100 -r requirements.txt
EXPOSE 80
#EXPOSE 8000
ENTRYPOINT ["python"] 
CMD ["main.py"]
