import json
from random import choices

from dataset_gen import DatasetGenerator
from helpers.vectorstore.faisser import FaissDB

if __name__ == "__main__":
    vectorstore = FaissDB(filename="functions.pkl")
    dg = DatasetGenerator(local_embeddings=True, validate=False, vectorstore=vectorstore)

    generate_n = 1000           # number of topics to generate for after the used ones
    multiplier = 1              # number of rows to generate for each topic
    num_processes = 10          # number of processes to use

    topics: list[str] = json.load(open("yahoo_questions_1.4M.json"))
    topics = choices(topics, k=generate_n)

    rows = dg.generate_parallel(topics, multiplier, num_processes)
    print(f"Generated {len(rows)} rows!")
