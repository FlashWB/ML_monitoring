import numpy
import matplotlib.pyplot as plt
import wordcloud
# from wordcloud import WordCload


def readData(path,num):
    vocab = []
    with open(path,'r',encoding='utf-8') as f:
        for line in f.readlines()[:num]:
            vocab.append(line.strip()[0])
    return vocab

wordcloud.WordCloud(
    font_path="C:/Windows/Fonts/simhei.ttf",
    background_color='white',
    mask=color_mask,
    max_words=2,
    max_font_size=40
)

vocab = readData('./FEATURE_ORDERED.txt',30)
print(vocab)

