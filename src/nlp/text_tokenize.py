from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer

from ML_Pipeline.Constants import max_text_length, vocab_size, model_dir
from ML_Pipeline.utils import save_tokenizer

oov_token = "<OOV>"
padding_type = "post"
trunction_type="post"

def build_tokenizer(df_train,num_words=None):
    if num_words is None:
        tokenizer = Tokenizer(oov_token=oov_token)
    else:
        tokenizer = Tokenizer(oov_token=oov_token,num_words=vocab_size)

    tokenizer.fit_on_texts(df_train)
    word_index = tokenizer.word_index
    print(" Word Index length ", len(word_index))
    print(" Number of Words:  ", tokenizer.num_words)
    save_tokenizer(tokenizer,num_words)
    return tokenizer

def prepare_seqence_data(text,tokenizer):

    print(text.head(2))
    # Create Sequence
    print(" Create Sequence ")
    text_sequences = tokenizer.texts_to_sequences(text)

    # Missing words in Glove vectors
    #words_used = [tokenizer.index_word[i] for i in range(1, vocab_size)]
    #missing_words = set(words_used) - set(word_vec.index.values)
    #print(len(missing_words))
    #missing_word_index = [tokenizer.word_index[word] for word in missing_words]


    # Pad the Sequences, because the sequences are not of the same length,
    # so let’s pad them to make them of similar length
    text_padded = pad_sequences(text_sequences, maxlen=max_text_length, padding=padding_type,
                                      truncating=trunction_type)

    # test_text_padded = pad_sequences(test_text_sequences, maxlen=max_text_length, padding=padding_type,
    #                                  truncating=trunction_type)

    print("Padded Sequence :: ", text_padded[0:5])

    return text_padded


