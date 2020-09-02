#coding=utf-8
__author__ = 'root'
import pandas as pd
import numpy as np
import pandas as pd
import wordcloud # 词云展示库
from PIL import Image # 图像处理库
import matplotlib.pyplot as plt

if __name__ == '__main__':
    word_dataframe = pd.read_csv('word.csv',encoding='gbk')
    
    image = Image.open('test.jpg')  # 作为背景轮廓图
    graph = np.array(image)
    
    wc = wordcloud.WordCloud(font_path='./fonts/simhei.ttf',background_color='black', max_words=100, mask=graph)
    
    name = list(word_dataframe.name)  # 词
    value = list(word_dataframe.frequency)  # 词的频率
    
    for i in range(len(name)):
        name[i] = str(name[i])
    dic = dict(zip(name, value))
    wc.generate_from_frequencies(dic)
    image_colors = wordcloud.ImageColorGenerator(graph) # 从背景图建立颜色方案
    wc.recolor(color_func=image_colors)
    plt.imshow(wc)
    plt.axis("off")  # 不显示坐标轴
    plt.show()
    wc.to_file('TEST.png')



            
