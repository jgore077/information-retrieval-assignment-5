from ranx import Qrels,Run,evaluate,compare
import sys
import os

METRICS=['precision@1','precision@5','ndcg@5','mrr','map']
EVAL_DIR="evals/"
qrel_path=sys.argv[1]
results_dir=sys.argv[2]
qrel=Qrels.from_file(qrel_path,kind="trec")

report = compare(
    qrels=Qrels.from_file(qrel_path,kind="trec"),
    runs=[Run.from_file(os.path.join(results_dir,file_path),kind="trec") for file_path in os.listdir(results_dir)],
    metrics=["precision@10"],
    max_p=0.05  # P-value threshold
)

frame=report.to_dataframe().sort_values("precision@10",ascending=False)
print(frame)