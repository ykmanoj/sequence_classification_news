import atexit
import json
from datetime import date

import pandas as pd
import numpy as np

from flask import *
import markdown
from apscheduler.schedulers.background import BackgroundScheduler

from markupsafe import Markup
from pandas.core.common import flatten

from ML_Pipeline import dataset
from ML_Pipeline.Constants import vocab_size, input_dir, root_dir, output_dir, label_map
from ML_Pipeline.data_processing import clean_datasets
from ML_Pipeline.dataset import read_json_request, read_data
from ML_Pipeline.utils import load_tokenizer, load_trained_model
from engine import run_training
from nlp.preprocess_text import merge_text_features, preparing_datasets
from nlp.text_tokenize import prepare_seqence_data


def app_initialization():
    app = Flask(__name__)
    # To avoid caching
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    # Raise exceptions from background tasks
    app.config['EXECUTOR_PROPAGATE_EXCEPTIONS'] = True

    model = load_trained_model('final_model','GRU')
    tokenized = load_tokenizer(vocab_size)


    @app.route('/', methods=['GET'])
    @app.route('/index/', methods=['GET'])
    def index():
        """
            Returns the read me file
        :return:
        """
        with open(root_dir+"/README") as f:
            content = f.read()
        content = Markup(markdown.markdown(content))
        return content

    @app.route('/start_training',methods=['GET'])
    def start_training():
        error = run_training()
        if error==0:
            error_dict = {"Code": error, "MSG": "Training completed successfully"}
        else:
            error_dict = {"Code": 0, "MSG": "Train failed"}

        return json.dumps(error_dict)

    @app.route('/single_news_prediction',methods=['POST'])
    def json_fakenews_prediction():
        try:
            print(request.json)
            #data = dict(request)
            #request_df = read_json_request(request.json)
            request_df = pd.io.json.json_normalize(request.get_json())
            request_df = clean_datasets(request_df)
            print(request_df.head())
            request_df = merge_text_features(request_df)
            request_df = preparing_datasets(request_df)
            tokenized = load_tokenizer()
            news_seq = prepare_seqence_data(request_df,tokenized)
            labels = pd.DataFrame(model.predict_classes(news_seq)).values
            print("News Type prediction::", labels)
            label = ['True' if l==0 else 'Fake' for l in labels]
            return json.dumps({'msg':'Success','label':label})
        except Exception as e:
            return json.dumps({'msg':str(e),'label':'error'})

    @app.route('/news_prediction', methods=['POST'])
    def fakenews_prediction():
        if True:
            path=str(request.data,encoding='utf8')
            print(" Fake News prediction for given path...", path)
            if path is None:
                data = read_data(r'test/test.csv')
            else:
                print("Path to read csv..",path)
                data = pd.read_csv(path)
            data = clean_datasets(data)
            print(data.head())
            request_df = merge_text_features(data)
            request_df = preparing_datasets(request_df)
            news_seq = prepare_seqence_data(request_df, tokenized)
            y_proba = model.predict(news_seq)
            #print(y_proba)
            resp_df =data.copy()
            label_pred = pd.DataFrame(y_proba,columns=['pred_prob'])
            resp_df['pred_prob'] = label_pred.pred_prob
            resp_df['label'] = np.round(label_pred.pred_prob)
            #resp_df['pred'] = resp_df[['pred']].apply(lambda x: np.round(x, decimals=2))
            print(resp_df[['pred_prob','label']].head())
            resp_df['label'].replace(label_map,inplace=True)
            output = output_dir+str(date.today())+'_prediction.csv'
            resp_df.to_csv(output)
            return json.dumps({'msg': 'Success and result saved to '+ output,
                               'predicted_label':resp_df['label'].tolist()})
        else:#except Exception as e:
            return json.dumps({'msg': str(e)})

    def all_schedular_jobs():
        try:
            data = dataset.read_directory(input_dir+'test/')
            request_df = merge_text_features(data)
            request_df = preparing_datasets(request_df)
            news_seq = prepare_seqence_data(request_df, tokenized)
            label = model.predict(news_seq)
            return json.dumps({'msg': 'Success', label: label})
        except Exception as e:
            return json.dumps({'msg': str(e), label: label})

    # scheduling jobs
    scheduler = BackgroundScheduler()
    scheduler.add_job(func=all_schedular_jobs, trigger="interval", hours=1 * 24)
    # scheduler.add_job(func=current_fleet_prediction_json, trigger="interval", hours=1*24)
    scheduler.start()
    # Shut down the scheduler when exiting the app
    atexit.register(lambda: scheduler.shutdown())
    return app


if __name__ == '__main__':
    app = app_initialization()
    app.run(debug=True, threaded=True, host='0.0.0.0',port=8080)
