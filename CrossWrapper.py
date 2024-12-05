from sentence_transformers import CrossEncoder
from SentenceTransformersWrapper import remove_html_tags,cosine_sim
from tqdm import tqdm
import torch
import json

def process_answers(answers:dict)->dict:
    new_answers={}
    for answer in answers:
        new_answers[answer["Id"]]=remove_html_tags(answer["Text"])

    return new_answers

class CrossEncoderWrapper():
    def __init__(self,answers_path,model_name="cross-encoder/ms-marco-MiniLM-L-12-v2") -> None:
        self.model_name=model_name
        self.answers_path=answers_path
        self.model=CrossEncoder(model_name,default_activation_function=torch.nn.Sigmoid())
        with open(answers_path,'r',encoding='utf-8') as answersfile:
            self.answers=process_answers(json.loads(answersfile.read()))
    
    
    def rerank(self,results:dict[str,float],query:str):
        reranked={}
        for id in results:
            cross_encoded=self.model.predict((query,self.answers[id]))
            reranked[id]=cross_encoded

        # Sort the dictionary so that the best answers are at the top of the dictionary
        reranked=dict(sorted(reranked.items(), key=lambda x: x[1],reverse=True))
        
        
        return reranked