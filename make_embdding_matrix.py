from util.utils import make_embedding_weight
from dataset import Dataset


def main():
    dataset = Dataset()
    dataset.set_config(base_dir='/Users/KaitoHH/Downloads')
    make_embedding_weight(dataset.tokenizer)


if __name__ == '__main__':
    main()
