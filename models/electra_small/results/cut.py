from gensim.models import word2vec
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import numpy as np
import pandas as pd
np.set_printoptions(threshold=np.inf)
#读取文件
test_df = pd.read_csv("test.csv")
train_df = pd.read_csv("train.csv")
dev_df = pd.read_csv("dev.csv")
#设置数值
embed_size = 200 # how big is each word vector
max_features = 800 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 128 # max number of words in a question to use

train_X = train_df["x_train"].values
val_X = dev_df["x_valid"].values
test_X = test_df["x_test"].values

## Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)
val_X = tokenizer.texts_to_sequences(val_X)
test_X = tokenizer.texts_to_sequences(test_X)

## Pad the sentences
train_X = pad_sequences(train_X, maxlen=maxlen)
val_X = pad_sequences(val_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)

#train_y = train_df['label'].values
#val_y = dev_df['label'].values
#test_y = test_df['label'].values


word_index = tokenizer.word_index
nb_words = len(word_index)  # 200593
print(nb_words)

train_y = pd.get_dummies(train_df['label']).values
val_y = pd.get_dummies(dev_df['label']).values
test_y = pd.get_dummies(test_df['label']).values

word_index = tokenizer.word_index
nb_words = len(word_index)  # 200593
print(nb_words)

print(train_X.shape)
print(val_X.shape)
print(test_X.shape)
print(train_y.shape)
print(val_y.shape)
print(test_y.shape)


#BiLSTM
def BiLSTM_model(test_X, test_y):

    model = joblib.load("senticorp_results.pkl")
    #评价模型
    result = model.predict(test_X)  # 预测样本属于每个类别的概率
    y_pred = result.argmax(axis=1)
    Y_test = test_y.argmax(axis=1)
    print('accuracy %s' % accuracy_score(y_pred, Y_test))
    target_names = ['1', '2', '3', '4', '5']
    print(classification_report(Y_test, y_pred, target_names=target_names))

BiLSTM_model( test_X, test_y)






