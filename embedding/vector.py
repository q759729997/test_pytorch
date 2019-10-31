import torchtext


def chinese_tokenizer(x):
    # 分词器
    return list(x)


if __name__ == "__main__":
    """
    https://torchtext.readthedocs.io/en/latest/data.html#fields
    """
    texts = ['我爱北京天安门', '兵者诡道也']
    # 构建Field对象
    data_text = torchtext.data.Field(
        sequential=True,
        use_vocab=True,
        fix_length=20,
        lower=True,
        tokenize=chinese_tokenizer)
    data_label = torchtext.data.Field(
        sequential=False, use_vocab=True, lower=True)
    # 构建vocab
    data_text.build_vocab(texts)
    print(data_text.vocab)
    print(len(data_text.vocab))  # 14
    # pad
    # padded = data_text.pad(texts)
    # print(padded)
