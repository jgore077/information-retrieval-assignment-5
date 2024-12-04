from SentenceTransformersWrapper import title_body_query
from ranx import Run
import json
import sys

def reformat_topics(topics):
    new_topics={}
    for topic in topics:
        id=topic["Id"]
        del topic["Id"]
        new_topics[id]=topic
    return new_topics

results_path=sys.argv[1]
topics_path=sys.argv[2]

result_dict=Run.from_file(results_path,kind="trec").to_dict()


with open(topics_path,encoding="utf-8") as topics_file:
    topics=reformat_topics(json.load(topics_file))
    

