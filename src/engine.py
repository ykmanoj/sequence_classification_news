############ Engine ################
    #   Run full application  #
    #   Extract datasets      #
    #   Prepare Data          #
    #   Training Datasets     #
    #   Model Evaluation      #
#####################################
from sklearn.model_selection import train_test_split

from ML_Pipeline import dataset
from ML_Pipeline.Constants import root_dir, dash, train_data_filename, \
    test_data_filename, test_label_file, vocab_size
from ML_Pipeline.explorative_analysis import check_dist
from ML_Pipeline.utils import performance_report, store_model, model_learning_history, load_trained_model, \
    load_tokenizer
from nlp.preprocess_text import preparing_datasets, process_labels, merge_text_features
from nlp.text_tokenize import prepare_seqence_data, build_tokenizer
from nlp.word_embeding import build_embeddings
from seq_model.build_network import build_network_lstm, train_model, build_network_GRU, \
    build_network_RNN

def run_training():
    print(root_dir)
    train = dataset.read_data(train_data_filename)
    test = dataset.read_data(test_data_filename)
    test_label = dataset.read_data(test_label_file)
    test['label'] = test_label.label

    column_names = train.columns
    print(column_names)
    train = dataset.clean_datasets(train)
    test = dataset.clean_datasets(test)

    print("Train data counts::", train.shape)
    print("Test data counts::", test.shape)

    check_dist(train, "fake_news_count_plot")

    # Merge multiple text columns like Title and Text
    train = merge_text_features(train)
    test = merge_text_features(test)

    X_train = preparing_datasets(train)
    X_test = preparing_datasets(test)

    y_train = process_labels(train.label)
    y_test = process_labels(test.label)

    print("Train Datasets after cleaning")
    print(X_train.head())

    print("test Datasets after cleaning")
    print(X_test.head())

    # Split for train and validation
    X_train,X_test,y_train,y_test = train_test_split(X_train,y_train,test_size=0.2)

    # Parameters ##########################
    vocab_size = 150000
    max_text_length = 100
    emb_dim = 100
    embedding_type = 'glove'
    epochs = 30
    batch_size = 256
    ##################################

    #tokenizer = build_tokenizer(X_train,num_words=vocab_size)
    tokenizer = build_tokenizer(X_train,vocab_size)
    train_text_seq = prepare_seqence_data(X_train, tokenizer)
    test_text_seq = prepare_seqence_data(X_test, tokenizer)

    print(" Tokenizer detail :: ", tokenizer.document_count)
    print('Vocabulary size:', len(tokenizer.word_counts))
    print('Shape of data padded:', train_text_seq.shape)

    ## Build Embedding layer
    embeding_layer = build_embeddings(tokenizer)
    model_type = 'GRU'
    name = 'experiment_final'
    ## Build Network
    if model_type == 'RNN':
        model = build_network_RNN(embeding_layer)
    elif model_type == 'GRU':
        model = build_network_GRU(embeding_layer)
    else:
        model = build_network_lstm(embeding_layer)


    name = "Model_" + model_type + str(epochs) + dash + str(batch_size) + dash + str(max_text_length) + dash + str(
        vocab_size) + dash

    model, history = train_model(model, train_text_seq, y_train, test_text_seq, y_test)
    model_learning_history(history, name)
    performance_report(model,model_type,name,test_text_seq, y_test)
    store_model(model, model_type,name)


if __name__ == '__main__':
    run_training()

