import os

import io
from datetime import date
import json
import pathlib
from zipfile import ZipFile
import wget
from os.path import exists

from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.python.keras.models import model_from_json, load_model
from tensorflow.python.keras.preprocessing.text import tokenizer_from_json

from ML_Pipeline.Constants import output_dir, model_dir, resource_dir, image_dir


# Encode features
def encode_features(train,features):
    print(" Encoding Features ")
    lbl_enc = preprocessing.OneHotEncoder()
    for col in features:
        lbl_enc.fit(train[[col]].values)
        train = lbl_enc.transform(train[[col]].values)
        #test = lbl_enc.transform(test[[col]].values)
        train.drop(col,axis=1,inplace=True)
        #test.drop(col,axis=1,inplace=True)
        print(train.columns)
        print(train.shape)

    return train

# Drop null record
def null_processing(feature_df):
    print("No of record with null values::", feature_df.isnull().sum() )
    columns = (feature_df.columns[feature_df.isnull().sum() > 0])
    print("Column having null values:: ", columns)
    feature_df.dropna(axis=0,inplace=True)

    return feature_df

def model_learning_history(history, name):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    #plt.show()
    plt.savefig(output_dir + image_dir + name + "_performance.png")

## Extract glove if not unzip
## And if not exist than download
def extractGlovefile():
    glove_dir = resource_dir+'glove/'
    #if ~pathlib.Path(glove_dir).exists():
    os.makedirs(glove_dir,exist_ok=True)
    file_zip = pathlib.Path(glove_dir + "glove.6B.zip")
    if file_zip.exists():
        with ZipFile(glove_dir + 'glove.6B.zip', 'r') as zip:
            # printing all the contents of the zip file
            zip.printdir()
            # extracting all the files
            print('Extracting all the files now...')
            zip.extractall(glove_dir)
            print('Done!')
    else:
        print("glove pretrained model not exists..downloading start..", )

        wget.download('http://nlp.stanford.edu/data/glove.6B.zip', out=glove_dir)
        # opening the zip file in READ mode
        with ZipFile(glove_dir + 'glove.6B.zip', 'r') as zip:
            # printing all the contents of the zip file
            zip.printdir()
            # extracting all the files
            print('Extracting all the files now...')
            zip.extractall(glove_dir)
            print('Done!')

def store_model(model,model_type, name):
    # Store the model as json and
    # store model weights as HDF5

    # serialize model to JSON
    model_json = model.to_json()
    with open(model_dir+model_type + name + "_model.json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights(model_dir+model_type + name + "_model.h5")

    model.save_weights()
    print("Saved model to disk")

def performance_report(model,name,model_type,testX,testy):
    import pandas as pd
    time = date.today()

    yhat_probs = model.predict(testX, verbose=0)
    # predict crisp classes for test set
    yhat_classes = model.predict_classes(testX, verbose=0)

    # reduce to 1d array
    yhat_probs = yhat_probs[:, 0]
    yhat_classes = yhat_classes[:, 0]

    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(testy, yhat_classes)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(testy, yhat_classes)
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(testy, yhat_classes)
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(testy, yhat_classes)
    print('F1 score: %f' % f1)

    if exists(model_dir + 'report.csv'):
        total_cost_df = pd.read_csv(model_dir + 'final.csv', index_col=0)
    else:
        total_cost_df = pd.DataFrame(
                columns=['time','model_type', 'name', 'Precision', 'Recall', 'f1_score', 'accuracy'])

    total_cost_df = total_cost_df.append(
            {'time': time,'model_type':model_type, 'name': name,'Precision': precision, 'Recall': recall, 'f1_score': f1,'accuracy':accuracy},
            ignore_index=True)
    total_cost_df.to_csv(output_dir + 'report.csv')

def load_trained_model(name,model_type):
    filename = model_dir + model_type + '/'+name
    # load json and create model
    json_file = open(filename+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(filename+".h5")
    print("Loaded model from disk")
    # summarize model.
    loaded_model.summary()
    # load dataset
    return loaded_model

def save_tokenizer(tokenizer,num_words):
    with io.open(model_dir + 'tokenizer_'+str(num_words)+'.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer.to_json(), ensure_ascii=False))
    f.close()
    return

def load_tokenizer(num_words=150000):
    with open(model_dir+'tokenizer_'+str(num_words)+'.json') as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)
    f.close()
    return tokenizer