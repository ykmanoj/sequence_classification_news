from pathlib import Path

root_dir = str(Path(__file__).parent.parent.parent)

input_dir = root_dir+"/input/"
output_dir = root_dir+"/output/"
model_dir = root_dir+"/model/"
image_dir = model_dir+'/images/'
resource_dir = root_dir+'/resource/'

train_data_filename='fake-news/train.csv'
test_data_filename='fake-news/test.csv'
test_label_file = 'fake-news/submit.csv'

column_names = ['id', 'title', 'author', 'text', 'label']

remove_columns = ['id','author']
categorical_features = []
target_col = ['label']
text_features = ['title', 'text']

label_map = {0:'True',1:'Fake'}

##########################
## Embedding Parametrs ###
vocab_size = 150000

max_text_length = 100
emb_dim = 100
embedding_type = 'glove'
model_type = 'LSTM'
#####################
###  HYPERPARAMS  ###
#####################
epochs = 30
batch_size = 256

lstm_size = 200
gru_size = 100
dense_layer_dim_1 = 32


dash = "-"
classifier = 'binary'

###### Flask App #########
trained_model = 'final_gru'

