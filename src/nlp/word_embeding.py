import pathlib

import numpy as np
import pandas as pd

from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import one_hot

from ML_Pipeline.Constants import max_text_length, embedding_type, emb_dim, vocab_size, root_dir, resource_dir
from ML_Pipeline.utils import extractGlovefile


##### Create GLove Word embedding #########
#### Download glove if not exist ########
def read_glove_embedings():
    glove_dir = resource_dir+'glove/'
    file = pathlib.Path(glove_dir + "glove.6B."+str(emb_dim)+"d.txt")
    if file.exists():
        #f = open(glove_dir + 'glove.6B.'+str(emb_dim)+'d.txt')
        print("glove pretrained model exists..", file.name)
    else:
        print("glove pretrained model not exists..")
        extractGlovefile()

    word_vec = pd.read_table(glove_dir+"glove.6B."+str(emb_dim)+"d.txt", sep=r"\s", header=None, engine='python',
                             encoding='iso-8859-1', error_bad_lines=False)
    word_vec.set_index(0, inplace=True)

    print('Found %s word vectors.' % len(word_vec))
    print('politics',word_vec.head())

    return word_vec

def glove_embedings(tokenizer):
    embeddings_index = read_glove_embedings()
    embedding_matrix = np.zeros((len(tokenizer.word_index)+1, emb_dim))

    index_n_word = [(i, tokenizer.index_word[i]) for i in range(1, len(embedding_matrix)) if
                    tokenizer.index_word[i] in embeddings_index.index]
    idx, word = zip(*index_n_word)
    embedding_matrix[idx, :] = embeddings_index.loc[word, :].values
    return embedding_matrix

def fasttext_embedings():
    pass

def onehot_embedding(tokenizer):
    onehot_vec =  [one_hot(words, (len(tokenizer.word_counts) +1)) for words in tokenizer.word_index.keys()]
    embedded_docs = pad_sequences(onehot_vec, padding='pre', maxlen=max_text_length)
    return embedded_docs

def build_embeddings(tokenizer):

    vocab_len = len(tokenizer.word_index)+1
    if embedding_type=='glove':
        embedding_matrix =  glove_embedings(tokenizer)
        embeddingLayer = Embedding(vocab_len, emb_dim,
                                   weights=[embedding_matrix], trainable=False)
    elif embedding_type=='fasttext':
        embedding_matrix =  fasttext_embedings()
        embeddingLayer = Embedding(input_dim=vocab_len, output_dim=emb_dim, input_length=max_text_length,
                                   weights=[embedding_matrix], trainable=False)
    else:
        embedding_matrix = onehot_embedding(tokenizer)
        embeddingLayer = Embedding(input_dim=vocab_len, output_dim=emb_dim, input_length=max_text_length,
                                   trainable=False)

    return embeddingLayer

