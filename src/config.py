vocab_size = -1  # 100000 #132779
vocab_file = None
unigram_prob_file = None

max_seq_len = 20

data_dir = '../data/definitions/'
embeddings_dir = '../data/embeddings/'

universal_embeddings = 'glove.840B.300d.txt'  # GoogleNews-vectors-negative300.bin
filtered_embeddings = 'glove_vocab.840B.300d.' + str(vocab_size) + '.pkl'
universal_embeddings_path = embeddings_dir + universal_embeddings
filtered_embeddings_path = embeddings_dir + filtered_embeddings

data_files = {
    'train': 'train.definitions.ids',
    'dev': 'dev.definitions.ids',
    'test': ['WN_seen_correct.txt', 'WN_unseen_correct.txt', 'concept_descriptions.txt', 'one_word_crossword.txt',
             'short_crossword.txt'],
}

tests = {
    'reverse-dictionary': {
        'seen': {'file_name': 'WN_seen_correct.txt', 'const_a': [0.001, 0.0003, 0.0001], 'normalize': False, 'num_pc': 1},
        'unseen': {'file_name': 'WN_unseen_correct.txt', 'const_a': [0.001, 0.0003, 0.0001], 'normalize': False, 'num_pc': 1},
        'made': {'file_name': 'concept_descriptions.txt', 'const_a': [0.001, 0.0003, 0.0001], 'normalize': False, 'num_pc': 1}
    },
    'crossword': {
        # 'one': {'file_name': 'one_word_crossword.txt', 'const_a': [0.001, 0.0003, 0.0001], 'normalize': False, 'num_pc': 1},
        'short': {'file_name': 'short_crossword.txt', 'const_a': [0.0003], 'normalize': False, 'num_pc': 1}
    }
}

# test set config without SVD
# tests = {
#     'seen': {'file_name': 'WN_seen_correct.txt', 'const_a': 0.000225, 'normalize': False, 'num_pc': 0},
#     # 'unseen': {'file_name': 'WN_unseen_correct.txt', 'const_a': 0.00003, 'normalize': True, 'num_pc': 1},
#     'made': {'file_name': 'concept_descriptions.txt', 'const_a': 0.000225, 'normalize': False, 'num_pc': 0}
# }
