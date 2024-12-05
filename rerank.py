from SentenceTransformersWrapper import title_body_query
from CrossWrapper import CrossEncoderWrapper
from tqdm import tqdm
from ranx import Run
import json
import sys
import os

def reformat_topics(topics):
    new_topics={}
    for topic in topics:
        id=topic["Id"]
        del topic["Id"]
        new_topics[id]=topic
    return new_topics

results_path=sys.argv[1]
topics_path=sys.argv[2]
output_file=sys.argv[3]

result_dict=Run.from_file(results_path,kind="trec").to_dict()

with open(topics_path,encoding="utf-8") as topics_file:
    topics=reformat_topics(json.load(topics_file))

cross_encoder=CrossEncoderWrapper('./data/Answers.json')

reranked_dict={}
for key in tqdm(result_dict,desc="Re-ranking with a cross-encoder"):
    reranked_dict[key]=cross_encoder.rerank(result_dict[key],title_body_query(topics[key]))
    
tmp_out_path=output_file+".trec"

Run(reranked_dict,name="cross-encoder-re-ranking").save(tmp_out_path)

os.rename(tmp_out_path,output_file)
