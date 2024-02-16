from tqdm import tqdm
from dataset_gen import DatasetGenerator

dg = DatasetGenerator(local_embeddings=True)

if __name__ == "__main__":
    for _ in tqdm(range(10)):
        dg.generate_auto(10, multiplier=2)
