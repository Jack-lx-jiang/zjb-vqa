from util.utils import make_embedding_weight
from dataset import Dataset

dataset = Dataset()
make_embedding_weight(dataset.tokenizer)
