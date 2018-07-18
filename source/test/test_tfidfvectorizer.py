import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
with open('./test','r',encoding='utf-8') as f:
    X = []
    Y = []
    for line in f.readlines()[:10]:
        try:
            sentence, label = line.strip().split('\t') 
        except:
            continue
        X.append(sentence)
        Y.append(label)
print(X)
Tfid = TfidfVectorizer(stop_words=None,min_df=1)
tf = CountVectorizer(stop_words=None)
data = Tfid.fit_transform(X)
# print(data)
# data2 = tf.fit_transform(X)
print(Tfid.get_feature_names)
# dictword = Tfid.get_feature_names
# print(dictword)








