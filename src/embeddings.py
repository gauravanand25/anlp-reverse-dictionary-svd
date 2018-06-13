import numpy as np
import gensim
import config
import pickle


class Embeddings:
    def __init__(self):
        self.embedding_len = None

    def normalize(self, word_vec):
        word_vec[2:] /= np.linalg.norm(word_vec[2:], axis=1).reshape(-1, 1)
        word_vec[0] = np.zeros(300)
        word_vec[1] = np.zeros(300)
        return word_vec

    def load_universal_embeddings(self):
        '''
        It can be either 'GLoVe' or 'Word2Vec'. Defined in config.
        :return:
        '''
        word_to_vec = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(config.universal_embeddings_path)
        self.embedding_len = word_to_vec.vector_size
        return word_to_vec

    def all_words(self, word_to_vec):
        return set(list(word_to_vec.vocab))

    def filter_vocab_embeddings(self, word_to_vec, vocab):
        '''
        Filter out the vectors for words in vocab list.
        :param word_to_vec: word vectors; either 'GLoVe' or "Word2Vec'.
        :param vocab: words in the vocab list are fetched from word_to_vec
        :return:
        '''

        universal_words = set(list(word_to_vec.vocab))
        vocab_to_vec = {}

        # verifying
        result = word_to_vec.most_similar(positive=['woman', 'king'], negative=['man'], topn=5)
        print(result)

        for idx, word in enumerate(vocab):
            if idx % 10000 == 0:
                print (idx, "/", len(vocab), "done")
            if word in universal_words:
                vocab_to_vec[word] = word_to_vec[word]
            else:
                print ("Word not found:", word)
        return vocab_to_vec

    def load_filtered_pretrained_embeddings(self, embeddings_file_path):
        """Loads pre-trained word embeddings.

        Args:
          embeddings_file_path: path to the pickle file with the embeddings.

        Returns:
          tuple of (dictionary of embeddings, length of each embedding).
        """
        print("Loading pretrained embeddings from %s" % embeddings_file_path)
        with open(embeddings_file_path, "rb") as input_file:
            pre_embs_dict = pickle.load(input_file)
        iter_keys = iter(pre_embs_dict.keys())
        first_key = next(iter_keys)
        embedding_length = len(pre_embs_dict[first_key])
        print("%d embeddings loaded; each embedding is length %d" %
              (len(pre_embs_dict.values()), embedding_length))

        self.embedding_len = embedding_length
        return pre_embs_dict, embedding_length

    def get_embedding_matrix(self, embedding_dict, vocab, emb_dim):
        emb_matrix = np.zeros([len(vocab), emb_dim])
        for word, ii in vocab.items():
            if word in embedding_dict:
                emb_matrix[ii] = embedding_dict[word]
            else:
                # numpy zeros for _UNK and _PAD
                print("OOV word when building embedding matrix: ", word)
        return np.asarray(emb_matrix)
