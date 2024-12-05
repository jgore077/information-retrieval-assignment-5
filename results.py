from SentenceTransformersWrapper import Wrapper,title_body_query
from tqdm import tqdm
from ranx import Run
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
    output_file=RESULTS_PATH+model+"_"+topics_path.split('.')[0].split("_")[1]+".tsv"
    wrapper=Wrapper(answers_path,model)
    
   
    for topic in tqdm(topics,desc=f"Computing results for {model}"):
        results[topic["Id"]]=wrapper.search(title_body_query(topic))
    
   
    tmp_out_path=output_file+".trec"

    Run(results,name=model).save(tmp_out_path)

    os.rename(tmp_out_path,output_file)
    # Delete model files to save space (Sentence Transformers take a lot of space)
    shutil.rmtree(wrapper.cache_dir)