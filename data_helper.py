import numpy as np
import re
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(data_file_path, text_name, label_name):
    """
    Returns split sentences and labels.
    text_name: the name of feature column
    label_name: the name of label column
    """
    # Load data from files
    data = pd.read_csv(data_file_path, error_bad_lines=False, warn_bad_lines=False)
    x_text = data[text_name]
    y = data[label_name]
    # clean texts
    x_text = [clean_str(sent) for sent in x_text]
    y = pd.get_dummies(y)
    return x_text, y.values


def next_batch(num, data, labels):
    """
    Return a total of `num` random samples and labels.
    :param num: batch_size
    :param data: feature column
    :param labels: label column
    :return: x_batch, y_batch
    """
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


def get_batches(data, labels, batch_size):
    for batch_i in range(0, len(data)//batch_size):
        start_i = batch_i * batch_size
        data_batch = data[start_i: start_i+batch_size]
        label_batch = labels[start_i: start_i+batch_size]
        yield data_batch, label_batch


if __name__ == '__main__':
    text, label = load_data_and_labels('./data/rt-polarity.csv', 'comment_text', 'label')
    print(label)


