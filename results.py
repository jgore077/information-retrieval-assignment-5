from SentenceTransformersWrapper import Wrapper
import shutil
import json
import sys
import os

answers_path=sys.argv[1]
topics_path=sys.argv[2]



RESULTS_PATH="results/"
os.makedirs(RESULTS_PATH,exist_ok=True)

with open("models.txt",encoding="utf-8") as model_file:
    models=[model.strip() for model in model_file.readlines()]
    
with open(topics_path,encoding="utf-8") as topics_file:
    topics=json.load(topics_file)
    
for model in models:
    results={}
    wrapper=Wrapper(answers_path,model)
    
    for topic in topics:
        results[topic["Id"]]=wrapper.search(topic["Title"]+" "+topic["Body"])
    
    # Delete model files to save space (Sentence Transformers take a lot of space)
    shutil.rmtree(wrapper.cache_dir)