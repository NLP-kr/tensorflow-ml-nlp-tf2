"""
유사한 BI-LSTM 접근법입니다.
static_bidirectional_rnn을 활용하였습니다.
"""
# coding: utf-8



import sys
import tensorflow as tf
import numpy as np
import os
import pandas as pd

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer

from sklearn.model_selection import train_test_split

# from tensorflow.keras import backend as K

import json

tf.logging.set_verbosity(tf.logging.INFO)

# # Initial global var

# In[ ]:


## 미리 Global 변수를 지정하자. 파일 명, 파일 위치, 디렉토리 등이 있다.

DATA_IN_PATH = '../data_in/'
DATA_OUT_PATH = '../data_out/'

TRAIN_Q1_DATA_FILE = 'train_q1.npy'
TRAIN_Q2_DATA_FILE = 'train_q2.npy'
TRAIN_LABEL_DATA_FILE = 'train_label.npy'
NB_WORDS_DATA_FILE = 'data_configs.json'

# # Load Dataset

# In[ ]:


## 데이터를 불러오는 부분이다. 효과적인 데이터 불러오기를 위해, 미리 넘파이 형태로 저장시킨 데이터를 로드한다.

q1_data = np.load(open(DATA_IN_PATH + TRAIN_Q1_DATA_FILE, 'rb'))
q2_data = np.load(open(DATA_IN_PATH + TRAIN_Q2_DATA_FILE, 'rb'))
labels = np.load(open(DATA_IN_PATH + TRAIN_LABEL_DATA_FILE, 'rb'))
prepro_configs = None

with open(DATA_IN_PATH + NB_WORDS_DATA_FILE, 'r') as f:
    prepro_configs = json.load(f)

## 학습에 필요한 파라메터들에 대해서 지정하는 부분이다.

print("# of dataset: {}".format(len(labels)))

BATCH_SIZE = 4096
EPOCH = 20
HIDDEN = 64
BUFFER_SIZE = len(q1_data)

NUM_LAYERS = 3
DROPOUT_RATIO = 0.3

TEST_SPLIT = 0.1
RNG_SEED = 13371447
EMBEDDING_DIM = 128
MAX_SEQ_LEN = 31

# In[ ]:


VOCAB_SIZE = prepro_configs['vocab_size']


# # Split train and test dataset

# In[ ]:


q1_data_len = np.array([min(len(x), MAX_SEQ_LEN) for x in q1_data])
q2_data_len = np.array([min(len(x), MAX_SEQ_LEN) for x in q2_data])

# In[ ]:


## 데이터를 나누어 저장하자. sklearn의 train_test_split을 사용하면 유용하다. 하지만, 쿼라 데이터의 경우는
## 입력이 1개가 아니라 2개이다. 따라서, np.stack을 사용하여 두개를 하나로 쌓은다음 활용하여 분류한다.

X = np.stack((q1_data, q2_data), axis=1)
y = labels
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=TEST_SPLIT, random_state=RNG_SEED)

train_Q1 = train_X[:,0]
train_Q2 = train_X[:,1]
test_Q1 = test_X[:,0]
test_Q2 = test_X[:,1]


# In[ ]:


def rearrange(base, hypothesis, labels):
    features = {"base": base, "hypothesis": hypothesis}
    return features, labels

def train_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((train_Q1, train_Q2, train_y))
    dataset = dataset.shuffle(buffer_size=len(train_Q1))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(rearrange)
    dataset = dataset.repeat(EPOCH)
    iterator = dataset.make_one_shot_iterator()
    
    return iterator.get_next()

def eval_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((test_Q1, test_Q2, test_y))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(rearrange)
    iterator = dataset.make_one_shot_iterator()
    
    return iterator.get_next()

class ManDist(Layer):
    """
    Keras Custom Layer that calculates Manhattan Distance.
    """

    # initialize the layer, No need to include inputs parameter!
    def __init__(self, **kwargs):
        self.result = None
        super(ManDist, self).__init__(**kwargs)

    # input_shape will automatic collect input shapes to build layer
    def build(self, input_shape):
        super(ManDist, self).build(input_shape)

    # This is where the layer's logic lives.
    def call(self, x, **kwargs):
        self.result = K.exp(-K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True))
        return self.result

    # return output shape
    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)


def BiRNN(x, dropout, scope, hidden_units):
    n_hidden = hidden_units
    n_layers = 3
    # Prepare data shape to match `static_rnn` function requirements
    x = tf.unstack(tf.transpose(x, perm=[1, 0, 2]))
    print(x)
    # Define lstm cells with tensorflow
    # Forward direction cell
    with tf.name_scope("fw" + scope), tf.variable_scope("fw" + scope):
        stacked_rnn_fw = []
        for _ in range(n_layers):
            fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
            lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=dropout)
            stacked_rnn_fw.append(lstm_fw_cell)
        lstm_fw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)

    with tf.name_scope("bw" + scope), tf.variable_scope("bw" + scope):
        stacked_rnn_bw = []
        for _ in range(n_layers):
            bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
            lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=dropout)
            stacked_rnn_bw.append(lstm_bw_cell)
        lstm_bw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_bw, state_is_tuple=True)
    # Get lstm cell output

    with tf.name_scope("bw" + scope), tf.variable_scope("bw" + scope):
        outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, x, dtype=tf.float32)
    return outputs[-1]


# # Model setup

def Malstm(features, labels, mode):
        
    TRAIN = mode == tf.estimator.ModeKeys.TRAIN
    EVAL = mode == tf.estimator.ModeKeys.EVAL
    PREDICT = mode == tf.estimator.ModeKeys.PREDICT
            
    embedding = tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
    
    base_embedded_matrix = embedding(features['base'])
    hypothesis_embedded_matrix = embedding(features['hypothesis'])

    base_sementic_matrix = BiRNN(base_embedded_matrix, DROPOUT_RATIO, 'base', HIDDEN)
    hypothesis_sementic_matrix = BiRNN(hypothesis_embedded_matrix, DROPOUT_RATIO, 'hypothesis', HIDDEN)

    logit_layer = ManDist()([base_sementic_matrix, hypothesis_sementic_matrix])
    logit_layer = tf.squeeze(logit_layer, axis=-1)

    # self._ma_dist([q1_lstm, q2_lstm])

    # logit_layer = tf.exp(-tf.reduce_sum(tf.abs(base_sementic_matrix - hypothesis_sementic_matrix), axis=1, keepdims=True))
    # logit_layer = tf.squeeze(logit_layer, axis=-1)
    #
    if PREDICT:
        return tf.estimator.EstimatorSpec(
                  mode=mode,
                  predictions={
                      'is_duplicate':logit_layer
                  })
    
    #prediction 진행 시, None
    if labels is not None:
        labels = tf.to_float(labels)
    
#     loss = tf.reduce_mean(tf.keras.metrics.binary_crossentropy(y_true=labels, y_pred=logit_layer))
    loss = tf.losses.mean_squared_error(labels=labels, predictions=logit_layer)
#     loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(labels, logit_layer))
    
    if EVAL:
        accuracy = tf.metrics.accuracy(labels, tf.round(logit_layer))
        eval_metric_ops = {'acc': accuracy}
        return tf.estimator.EstimatorSpec(
                  mode=mode,
                  eval_metric_ops= eval_metric_ops,
                  loss=loss)

    elif TRAIN:

        global_step = tf.train.get_global_step()
        train_op = tf.train.AdamOptimizer(1e-3).minimize(loss, global_step)

        return tf.estimator.EstimatorSpec(
                  mode=mode,
                  train_op=train_op,
                  loss=loss)


# # Training & Eval

# In[ ]:


os.environ["CUDA_VISIBLE_DEVICES"]="7" #For TEST

model_dir = os.path.join(os.getcwd(), DATA_OUT_PATH + "/checkpoint/rnn2/")
os.makedirs(model_dir, exist_ok=True)

config_tf = tf.estimator.RunConfig()

lstm_est = tf.estimator.Estimator(Malstm, model_dir=model_dir)


# In[ ]:


lstm_est.train(train_input_fn)


# In[ ]:


lstm_est.evaluate(eval_input_fn)


# # Load test dataset |& create submit dataset to kaggle

# In[ ]:


TEST_Q1_DATA_FILE = 'test_q1.npy'
TEST_Q2_DATA_FILE = 'test_q2.npy'
TEST_ID_DATA_FILE = 'test_id.npy'

test_q1_data = np.load(open(DATA_IN_PATH + TEST_Q1_DATA_FILE, 'rb'))
test_q2_data = np.load(open(DATA_IN_PATH + TEST_Q2_DATA_FILE, 'rb'))
test_id_data = np.load(open(DATA_IN_PATH + TEST_ID_DATA_FILE, 'rb'))


# In[ ]:


predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"base":test_q1_data, 
                                                         "hypothesis":test_q2_data}, 
                                                      shuffle=False)

predictions = np.array([p['is_duplicate'] for p in lstm_est.predict(input_fn=
predict_input_fn)])


# In[ ]:


print(len(predictions)) #2345796

output = pd.DataFrame( data={"test_id":test_id_data, "is_duplicate": list(predictions)} )
output.to_csv( "rnn_predict.csv", index=False, quoting=3 )

