import torchtext


def chinese_tokenizer(x):
    # 分词器
    return list(x)


def get_dataset(txt_data, text_field, label_field):
    # 读取数据
    fields = [("id", None), ("feature", text_field), ("label", label_field)]
    examples = []
    for txt in txt_data:
        label, feature = txt.split('\t')
        examples.append(
            torchtext.data.Example.fromlist([None, feature, label], fields))
    return examples, fields


if __name__ == "__main__":
    """
    https://torchtext.readthedocs.io/en/latest/data.html#fields
    https://blog.csdn.net/nlpuser/article/details/88067167
    """
    texts = ['娱乐\t我爱北京天安门', '军事\t兵者诡道也']
    # 构建Field对象
    data_text = torchtext.data.Field(
        sequential=True,  # Whether the datatype represents sequential data. If False, no tokenization is applied
        use_vocab=True,  # Whether to use a Vocab object. If False, the data in this field should already be numerical
        fix_length=10,  # A fixed length that all examples using this field will be padded to, or None for flexible sequence lengths
        lower=True,
        tokenize=chinese_tokenizer)
    data_label = torchtext.data.Field(
        sequential=False, use_vocab=True, lower=True)
    # 构建数据集
    train_examples, train_fields = get_dataset(texts, data_text, data_label)
    train_data = torchtext.data.Dataset(train_examples, train_fields)
    # 构建迭代器，BucketIterator一次处理多个dataset，BucketIterator相比Iterator的优势是会自动选取样本长度相似的数据来构建批数据
    train_iter = torchtext.data.BucketIterator.splits(
        (train_data,),  # 构建数据集所需的数据集
        batch_sizes=(1,),
        device=-1,  # 如果使用gpu，此处将-1更换为GPU的编号
        sort_key=lambda x: len(x.feature),  # the BucketIterator needs to be told what function it should use to group the data.
        sort_within_batch=False,
        repeat=False  # we pass repeat=False because we want to wrap this Iterator layer.
    )[0]
    # 构建vocabulary
    data_text.build_vocab(train_data)
    data_label.build_vocab(train_data)
    # batch迭代
    for epoch, batch in enumerate(train_iter):
        print(batch.feature)
        print(batch.label)
