import tensorflow as tf
import numpy as np
import data_helper
from tensorflow.contrib import learn
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from tensorflow.contrib.rnn import BasicLSTMCell

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string('data_file_path', './data/rt-polarity.csv', 'Data source')
tf.flags.DEFINE_string('feature_name', 'comment_text', 'The name of feature column')
tf.flags.DEFINE_string('label_name', 'label', 'The name of label column')
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")

FLAGS = tf.flags.FLAGS


def tokenizer(docs):
    for doc in docs:
        yield doc.split(' ')


def pre_process():
    # load data
    x_text, y = data_helper.load_data_and_labels(FLAGS.data_file_path, FLAGS.feature_name, FLAGS.label_name)
    # Build vocabulary and cut or extend sentence to fixed length
    max_document_length = max([len(x) for x in tokenizer(x_text)])
    print('max document length: {}'.format(max_document_length))
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length, tokenizer_fn=tokenizer)
    # replace the word using the index of word in vocabulary
    x = np.array(list(vocab_processor.fit_transform(x_text)))

    # random shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    del x, y, x_shuffled, y_shuffled

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, vocab_processor, x_dev, y_dev


class ABLSTM(object):
    def __init__(self, config):
        self.max_len = config["max_len"]
        self.hidden_size = config["hidden_size"]
        self.vocab_size = config["vocab_size"]
        self.embedding_size = config["embedding_size"]
        self.n_class = config["n_class"]
        self.learning_rate = config["learning_rate"]

        # placeholder
        self.x = tf.placeholder(tf.int32, [None, self.max_len])
        self.label = tf.placeholder(tf.float32, [None, self.n_class])
        self.keep_prob = tf.placeholder(tf.float32)

    def build_graph(self):
        print("building graph")
        # Word embedding
        embeddings_var = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                                     trainable=True)
        batch_embedded = tf.nn.embedding_lookup(embeddings_var, self.x)

        rnn_outputs, _ = bi_rnn(BasicLSTMCell(self.hidden_size),
                                BasicLSTMCell(self.hidden_size),
                                inputs=batch_embedded, dtype=tf.float32)

        fw_outputs, bw_outputs = rnn_outputs

        W = tf.Variable(tf.random_normal([self.hidden_size], stddev=0.1))
        H = fw_outputs + bw_outputs  # (batch_size, seq_len, HIDDEN_SIZE)
        M = tf.tanh(H)  # M = tanh(H)  (batch_size, seq_len, HIDDEN_SIZE)

        self.alpha = tf.nn.softmax(tf.reshape(tf.matmul(tf.reshape(M, [-1, self.hidden_size]),
                                                        tf.reshape(W, [-1, 1])),
                                              (-1, self.max_len)))  # batch_size x seq_len
        r = tf.matmul(tf.transpose(H, [0, 2, 1]),
                      tf.reshape(self.alpha, [-1, self.max_len, 1]))
        r = tf.squeeze(r)
        h_star = tf.tanh(r)  # (batch , HIDDEN_SIZE

        h_drop = tf.nn.dropout(h_star, self.keep_prob)

        # Fully connected layer（dense layer)
        FC_W = tf.Variable(tf.truncated_normal([self.hidden_size, self.n_class], stddev=0.1))
        FC_b = tf.Variable(tf.constant(0., shape=[self.n_class]))
        y_hat = tf.nn.xw_plus_b(h_drop, FC_W, FC_b)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=self.label))

        # prediction
        self.prediction = tf.argmax(tf.nn.softmax(y_hat), 1)

        # optimization
        loss_to_minimize = self.loss
        tvars = tf.trainable_variables()
        gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
        grads, global_norm = tf.clip_by_global_norm(gradients, 1.0)

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step,
                                                       name='train_step')
        print("graph built successfully!")


if __name__ == '__main__':
    x_train, y_train, vocab_processor, x_dev, y_dev = pre_process()
    vocab_size = len(vocab_processor.vocabulary_)

    config = {
        "max_len": x_train.shape[1],
        "hidden_size": 64,
        "vocab_size": vocab_size,
        "embedding_size": 128,
        "n_class": y_train.shape[1],
        "learning_rate": 1e-3,
        "batch_size": 64,
        "train_epoch": 10
    }

    classifier = ABLSTM(config)
    classifier.build_graph()

    # accuracy
    correct_prediction = tf.equal(classifier.prediction, tf.argmax(classifier.label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    with tf.Session() as sess:
        # init variables
        init = tf.global_variables_initializer()
        sess.run(init)
        # train
        for epoch_i in range(config['train_epoch']):
            for batch_i, (x_batch, y_batch) in enumerate(data_helper.get_batches(x_train, y_train, config['batch_size'])):
                _, acc, loss = sess.run([classifier.train_op, accuracy, classifier.loss],
                                        feed_dict={classifier.x: x_batch, classifier.label: y_batch, classifier.keep_prob: 0.8})
                if batch_i % 10 == 0:
                    print('Epoch {}/{}, Batch {}/{}, loss: {}, accuracy: {}'.format(epoch_i, config['train_epoch'],
                                                                                    batch_i,
                                                                                    len(x_train) // config[
                                                                                        'batch_size'],
                                                                                    loss, acc))
        # valid step
        print('valid accuracy: {}'.format(sess.run(accuracy, feed_dict={classifier.x: x_dev, classifier.label: y_dev, classifier.keep_prob: 1.})))
