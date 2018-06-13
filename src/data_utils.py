import config as config
import re
import os
import tensorflow as tf
import collections
from embeddings import Embeddings
import pickle
import numpy as np

# Special vocabulary symbols - these get placed at the start of vocab.
_PAD = "_PAD"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _UNK]

PAD_ID = 0
UNK_ID = 1

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d")


def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(_WORD_SPLIT.split(space_separated_fragment))
    return [w.lower() for w in words if w]


def pad_sequence(sequence, max_seq_len):
    padding_required = max_seq_len - len(sequence)
    # Sentence too long, so truncate.
    if padding_required < 0:
        padded = sequence[:max_seq_len]
    # Sentence too short, so pad.
    else:
        padded = sequence + ([PAD_ID] * padding_required)
    return padded


def token_ids_to_sentence(token_ids, idx_to_word):
    token_ids = list(token_ids.reshape(-1))
    sent = []
    for token in token_ids:
        if token == 0:
            break
        sent.append(idx_to_word[token])
    return ' '.join(sent)


def sentence_to_token_ids(sentence, vocabulary, tokenizer=None,
                          normalize_digits=True):
    """Convert a string to list of integers representing token ids.

    For example, a sentence "I have a dog" may become tokenized into
    ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
    "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

    Args:
      sentence: the sentence in bytes format to convert to token-ids.
      vocabulary: a dictionary mapping tokens to integers.
      tokenizer: a function to use to tokenize each sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.

    Returns:
      a list of integers, the token-ids for the sentence.
    """
    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]
    else:
        # Normalize digits by 0 before looking words up in the vocabulary.
        return [vocabulary.get(_DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]


class Crossword:
    def __init__(self, dataset):
        self.dataset = dataset

    def data_to_token_ids(self, data_path, tokenizer=None):
        """Tokenize data file and turn into token-ids using given vocabulary file.

        Loads data line-by-line from data_path, calls sentence_to_token_ids.
        See sentence_to_token_ids on the details of token-ids format.
        Also pads out each sentence with the _PAD id, or truncates,
        so that each sentence is the same length.

        We also remove any instance of the head word from the corresponding
        gloss, since we don't want to define any word in terms of itself.

        Args:
          data_path: path to the data file in one-sentence-per-line format.
          target_path: path where the file with token-ids will be created.
          vocabulary_path: path to the vocabulary file.
          max_seq_len: maximum sentence length before truncation applied.
          tokenizer: a function to use to tokenize each sentence;
            if None, basic_tokenizer will be used.
          normalize_digits: Boolean; if true, all digits are replaced by 0s.
        """
        # Check if token-id version of the data exists; if so, do nothing.
        # print("Encoding data into token-ids in %s" % data_path)
        # vocab is a mapping from tokens to ids.
        # Write each data line to an id-based heads and glosses file.
        X, y, dig = [], [], []

        with tf.gfile.GFile(data_path, mode="r") as data_file:
            counter = 0
            for line in data_file:
                counter += 1
                if counter % 100000 == 0:
                    print("encoding training data line %d" % counter)

                token_ids = sentence_to_token_ids(
                    line, self.dataset.word_to_idx, tokenizer, normalize_digits=False)
                # tf.compat.as_bytes(line), vocab, tokenizer, normalize_digits)
                # Write out the head ids, one head per line.
                y.append(token_ids[0])
                dig.append(int(line.split()[-1]))
                # Remove all instances of head word in gloss.
                clean_gloss = [w for w in token_ids[1:-1] if w != token_ids[0]]
                # Pad out the glosses, or truncate, so all the same length.
                glosses_ids = pad_sequence(clean_gloss, config.max_seq_len)
                X.append(glosses_ids)
        return np.array(X), np.array(y), np.array(dig)


class ReverseDictionary:
    def __init__(self):
        if config.vocab_file is None:
            self.vocab_file = os.path.join(config.data_dir, "definitions_%s" % config.vocab_size)
            self.unigram_prob_file = os.path.join(config.data_dir, "definitions_%s" % config.vocab_size)
        else:
            self.vocab_file = config.vocab_file
            self.unigram_prob_file = config.unigram_prob_file

        self.total_vocab = None
        self.target_vocab = None
        self.input_vocab = None
        self.word_vec = None
        self.word_prob = None
        self.word_to_idx = None
        self.idx_to_word = None

    def prepare_dict_data(self, data_dir, train_file, dev_file, max_seq_len, tokenizer=None):
        """Get processed data into data_dir, create vocabulary.

        Args:
          data_dir: directory in which the data sets will be stored.
          train_file: file with dictionary definitions for training.
          dev_file: file with dictionary definitions for development testing.
          vocabulary_size: size of the vocabulary to create and use.
          max_seq_len: maximum sentence length before applying truncation.
          tokenizer: a function to use to tokenize each data sentence;
            if None, basic_tokenizer will be used.
        """
        train_path = os.path.join(data_dir, train_file)
        dev_path = os.path.join(data_dir, dev_file)

        # Create vocabulary of the appropriate size.
        vocab_path_stem = os.path.join(data_dir, "definitions_%d" % config.vocab_size)
        self.create_vocabulary(vocab_path_stem, tokenizer)

        # Create versions of the train and dev data with token ids.
        train_ids = os.path.join(
            data_dir, "train.definitions.ids")
        dev_ids = os.path.join(data_dir, "dev.definitions.ids")
        self.data_to_token_ids(train_path, tokenizer)
        self.data_to_token_ids(dev_path, tokenizer)

    def create_vocabulary(self, vocab_path_stem, embedding_words, tokenizer=None):
        """Create vocabulary files (if they not do exist) from data file.

        Assumes lines in the input data are: head_word gloss_word1
        gloss_word2... The glosses are tokenised, and the gloss vocab
        contains the most frequent tokens up to vocabulary_size. Vocab is
        written to vocabulary_path in a one-token-per-line format, so that
        the token in the first line gets id=0, second line gets id=1, and so
        on. A list of all head words is also written.

        We also assume that the final processed glosses won't contain any
        instances of the corresponding head word (since we don't want a word
        defined in terms of itself), and the vocab counts calculated here
        will reflect that.

        Args:
          vocab_path_stem: path where the vocab files will be created.
          data_path: data file that will be used to create vocabulary.
          vocabulary_size: limit on the size of the vocab (glosses and heads).
          tokenizer: a function to use to tokenize each data sentence;
            if None, basic_tokenizer will be used.
        """
        # Check if the vocabulary already exists; if so, do nothing.
        if not tf.gfile.Exists(vocab_path_stem + ".vocab"):
            print("Creating vocabulary %s from data" % (vocab_path_stem))
            # Counts for the head words.
            head = collections.defaultdict(int)
            # Counts for all words (heads and glosses).
            words = collections.defaultdict(int)
            counter = 0
            for file_type, files in config.data_files.items():
                for file_name in files:
                    data_path = os.path.join(config.data_dir, file_name)
                    with tf.gfile.GFile(data_path, mode="r") as f:
                        for line in f:
                            counter += 1
                            if counter % 100000 == 0:
                                print("  processing training data line %d" % counter)
                            # line = tf.compat.as_bytes(line)
                            tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)

                            words[tokens[0]] += 1
                            head[tokens[0]] += 1
                            for word in tokens[1:]:
                                # Assume final gloss won't contain the corresponding head.
                                if word != tokens[0]:
                                    words[word] += 1

            # Sort words by frequency, adding _PAD and _UNK to all_words.
            words = {word: cnt for word, cnt in words.items() if word in embedding_words}
            head = {head_word: cnt for head_word, cnt in head.items() if head_word in embedding_words}
            all_words = _START_VOCAB + sorted(words, key=words.get, reverse=True)
            head_vocab = sorted(head, key=head.get, reverse=True)

            # convert unigram frequency into probability
            total = sum(words.values())
            word_prob = {k: float(v) / total for k, v in words.items()}
            print("Writing out vocabulary")
            if config.vocab_size > 0:
                assert len(all_words) >= config.vocab_size, (
                    "vocab size must be less than %s, the total"
                    "no. of words in the training data" % len(all_words))
            # Write the head words to file.
            with tf.gfile.GFile(vocab_path_stem + "_head_words.txt", mode="w") as head_file:
                for w in head_vocab:
                    head_file.write(w + "\n")
            # Write all words to file.
            with tf.gfile.GFile(vocab_path_stem + ".vocab", mode="w") as vocab_file:
                for w in all_words[:config.vocab_size]:
                    vocab_file.write(w + "\n")
            # Write probability of all words to file
            with open(vocab_path_stem + ".prob.pkl", 'wb') as output_file:
                pickle.dump(word_prob, output_file, protocol=pickle.HIGHEST_PROTOCOL)
            print("Data pre-processing complete")

    def get_unigram_array(self, word_prob, word_to_idx, const_a):
        word_weight = np.zeros(len(word_to_idx))
        for word, weight in word_prob.items():
            if word in word_to_idx:
                word_weight[word_to_idx[word]] = const_a/(const_a + weight)
        return word_weight

    def read_vocabulary(self, vocabulary_path):
        """Initialize vocabulary from file vocabulary_path.

        We assume the vocabulary is stored one-item-per-line, so a file:
          dog
          cat
        will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
        also return the reversed-vocabulary ["dog", "cat"].

        Args:
          vocabulary_path: path to the file containing the vocabulary.

        Returns:
          a pair: the vocabulary (a dictionary mapping string to integers), and
          the reversed vocabulary (a list, which reverses the vocabulary mapping).

        Raises:
          ValueError: if the provided vocabulary_path does not exist.
        """
        if tf.gfile.Exists(vocabulary_path + ".vocab"):
            rev_vocab = []
            with tf.gfile.GFile(vocabulary_path + ".vocab", mode="r") as f:
                rev_vocab.extend(f.readlines())
            # rev_vocab = [tf.compat.as_bytes(line.strip()) for line in rev_vocab]
            rev_vocab = [line.strip() for line in rev_vocab]
            vocab = {x: y for (y, x) in enumerate(rev_vocab)}
            rev_vocab = {y: x for x, y in vocab.items()}
            return vocab, rev_vocab
        else:
            raise ValueError("Vocabulary file %s not found.", vocabulary_path)

    def read_unigram_freq(self, unigram_prob_path):
        probs_embs_dict = None
        if tf.gfile.Exists(unigram_prob_path + ".prob.pkl"):
            with open(unigram_prob_path + ".prob.pkl", "rb") as input_file:
                probs_embs_dict = pickle.load(input_file)
        return probs_embs_dict

    def sentence_to_token_ids(self, sentence, vocabulary, tokenizer=None,
                              normalize_digits=True):
        """Convert a string to list of integers representing token ids.

        For example, a sentence "I have a dog" may become tokenized into
        ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
        "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

        Args:
          sentence: the sentence in bytes format to convert to token-ids.
          vocabulary: a dictionary mapping tokens to integers.
          tokenizer: a function to use to tokenize each sentence;
            if None, basic_tokenizer will be used.
          normalize_digits: Boolean; if true, all digits are replaced by 0s.

        Returns:
          a list of integers, the token-ids for the sentence.
        """
        if tokenizer:
            words = tokenizer(sentence)
        else:
            words = basic_tokenizer(sentence)
        if not normalize_digits:
            return [vocabulary.get(w, UNK_ID) for w in words]
        else:
            # Normalize digits by 0 before looking words up in the vocabulary.
            return [vocabulary.get(_DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]

    def data_to_token_ids(self, data_path, tokenizer=None, normalize_digits=True):
        """Tokenize data file and turn into token-ids using given vocabulary file.

        Loads data line-by-line from data_path, calls sentence_to_token_ids.
        See sentence_to_token_ids on the details of token-ids format.
        Also pads out each sentence with the _PAD id, or truncates,
        so that each sentence is the same length.

        We also remove any instance of the head word from the corresponding
        gloss, since we don't want to define any word in terms of itself.

        Args:
          data_path: path to the data file in one-sentence-per-line format.
          target_path: path where the file with token-ids will be created.
          vocabulary_path: path to the vocabulary file.
          max_seq_len: maximum sentence length before truncation applied.
          tokenizer: a function to use to tokenize each sentence;
            if None, basic_tokenizer will be used.
          normalize_digits: Boolean; if true, all digits are replaced by 0s.
        """
        # Check if token-id version of the data exists; if so, do nothing.
        # print("Encoding data into token-ids in %s" % data_path)
        # vocab is a mapping from tokens to ids.
        # Write each data line to an id-based heads and glosses file.
        X, y = [], []

        with tf.gfile.GFile(data_path, mode="r") as data_file:
            counter = 0
            for line in data_file:
                counter += 1
                if counter % 100000 == 0:
                    print("encoding training data line %d" % counter)
                token_ids = self.sentence_to_token_ids(
                    line, self.word_to_idx, tokenizer, normalize_digits)
                # tf.compat.as_bytes(line), vocab, tokenizer, normalize_digits)
                # Write out the head ids, one head per line.
                y.append(token_ids[0])
                # Remove all instances of head word in gloss.
                clean_gloss = [w for w in token_ids[1:] if w != token_ids[0]]
                # Pad out the glosses, or truncate, so all the same length.
                glosses_ids = pad_sequence(clean_gloss, config.max_seq_len)
                X.append(glosses_ids)
        return np.array(X), np.array(y)

    def write_data_to_token_ids(self, data_path, target_path, max_seq_len=config.max_seq_len,
                          tokenizer=None, normalize_digits=True):
        """Tokenize data file and turn into token-ids using given vocabulary file.

        Loads data line-by-line from data_path, calls sentence_to_token_ids,
        and saves the result to target_path. See sentence_to_token_ids on
        the details of token-ids format. Also pads out each sentence with
        the _PAD id, or truncates, so that each sentence is the same length.

        We also remove any instance of the head word from the corresponding
        gloss, since we don't want to define any word in terms of itself.

        Args:
          data_path: path to the data file in one-sentence-per-line format.
          target_path: path where the file with token-ids will be created.
          vocabulary_path: path to the vocabulary file.
          max_seq_len: maximum sentence length before truncation applied.
          tokenizer: a function to use to tokenize each sentence;
            if None, basic_tokenizer will be used.
          normalize_digits: Boolean; if true, all digits are replaced by 0s.
        """
        # Check if token-id version of the data exists; if so, do nothing.
        if not (tf.gfile.Exists(target_path + ".gloss") and
                    tf.gfile.Exists(target_path + ".head")):
            print("Encoding data into token-ids in %s" % data_path)
            # vocab is a mapping from tokens to ids.
            vocab = self.word_to_idx
            # Write each data line to an id-based heads and glosses file.
            with tf.gfile.GFile(data_path, mode="r") as data_file:
                with tf.gfile.GFile(target_path + ".gloss", mode="w") as glosses_file:
                    with tf.gfile.GFile(target_path + ".head", mode="w") as heads_file:
                        counter = 0
                        for line in data_file:
                            counter += 1
                            if counter % 100000 == 0:
                                print("encoding training data line %d" % counter)
                            token_ids = self.sentence_to_token_ids(
                                line, vocab, tokenizer, normalize_digits)
                            # tf.compat.as_bytes(line), vocab, tokenizer, normalize_digits)
                            # Write out the head ids, one head per line.
                            heads_file.write(str(token_ids[0]) + "\n")
                            # Remove all instances of head word in gloss.
                            clean_gloss = [w for w in token_ids[1:] if w != token_ids[0]]
                            # Pad out the glosses, or truncate, so all the same length.
                            glosses_ids = pad_sequence(clean_gloss, max_seq_len)
                            # Write out the glosses as ids, one gloss per line.
                            glosses_file.write(" ".join([str(t) for t in glosses_ids]) + "\n")


    def load_embeddings_vocab(self):
        pretrained_embeddings = Embeddings()

        # read filtered embeddings
        if not tf.gfile.Exists(config.filtered_embeddings_path):
            word_to_vec = pretrained_embeddings.load_universal_embeddings()

            self.create_vocabulary(self.vocab_file, pretrained_embeddings.all_words(word_to_vec), tokenizer=None)
            word_to_idx, idx_to_word = self.read_vocabulary(self.vocab_file)

            filtered_embeddings = pretrained_embeddings.filter_vocab_embeddings(word_to_vec, word_to_idx.keys())

            with open(config.filtered_embeddings_path, 'wb') as output_file:
                pickle.dump(filtered_embeddings, output_file, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            word_to_idx, idx_to_word = self.read_vocabulary(self.vocab_file)
            word_prob = self.read_unigram_freq(self.unigram_prob_file)
            assert 1.01 > sum([0 if val is None else val for val in word_prob.values()]) > 0.99, "What?!"

        pre_embs_dict, embd_dim = pretrained_embeddings.load_filtered_pretrained_embeddings(config.filtered_embeddings_path)
        word_vec = pretrained_embeddings.get_embedding_matrix(pre_embs_dict, word_to_idx, embd_dim)

        self.word_vec = word_vec
        self.word_prob = word_prob
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word

        train_path = os.path.join(config.data_dir, config.data_files['train'])
        dev_path = os.path.join(config.data_dir, config.data_files['dev'])
        self.write_data_to_token_ids(train_path, target_path=train_path)
        self.write_data_to_token_ids(dev_path, target_path=dev_path)


if __name__ == '__main__':
    dataset = ReverseDictionary()
    dataset.load_embeddings_vocab()
    
