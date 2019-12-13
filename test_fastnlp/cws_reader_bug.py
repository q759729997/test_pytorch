from fastNLP.io import CWSPipe


if __name__ == "__main__":
    """cwsPipe读取语料bug"""
    pipe = CWSPipe(dataset_name=None)
    data = pipe.process_from_file(paths='data/test_bug.txt')
    print(data)
