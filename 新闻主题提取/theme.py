#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-17 09:08:11
# @Author  : Li Shunli (17801063098@163.com)
# @Link    : http://github.com/lishunli-notebook
# @Method  : LDA（Latent Dirichlet Allocation）

import os
import numpy as np
import jieba

"""
LSA(Latent Semantic Space 隐含语义空间)介绍：采用SVD降维，将document-term空间映射到潜义空间。
    优点：
        低维空间表示可以刻画同义词，同义词会对应着相同或相似的主题；
        降维可去除部分噪声，使特征更鲁棒(稳健)；
        充分利用冗余数据；
        无监督/完全自动化；
        与语言无关。
    缺点：
        没有刻画term出现次数的概率模型；
        无法解决多义词的问题；
        SVD的优化目标基于L-2 norm 或者是 Frobenius Norm的，这相当于隐含了对数据的高斯噪声假设。而term出现的次数是非负的，这明显不符合Gaussian假设，而更接近Multi-nomial分布；
        对于count vectors 而言，欧式距离表达是不合适的（重建时会产生负数）；
        特征向量的方向没有对应的物理解释；
        SVD的计算复杂度很高，而且当有新的文档来到时，若要更新模型需重新训练；
        维数的选择是ad-hoc的；

LDA（Latent Dirichlet Allocation）:主题模型。 #https://blog.csdn.net/v_JULY_v/article/details/41209515
语料库和词料库矩阵：n*m = n*K - K*m, (n个文档，m个词，通过训练，得到K个topic)
给定文章，反推作者写文章的主题。

PLSA和LDA的区别：
    PLSA中，主题分布和词分布是唯一确定的，能明确的指出主题分布可能就是{教育：0.5，经济：0.3，交通：0.2}，
    词分布可能就是{大学：0.5，老师：0.3，课程：0.2}。
    但在LDA中，主题分布和词分布不再唯一确定不变，即无法确切给出。例如主题分布可能是{教育：0.5，经济：0.3，交通：0.2}，
    也可能是{教育：0.6，经济：0.2，交通：0.2}，到底是哪个我们不再确定（即不知道），因为它是随机的可变化的。
    但再怎么变化，也依然服从一定的分布，即主题分布跟词分布由Dirichlet先验随机确定。
"""
class themeAnalysis(object):
    """
    LSA/LDA
    """
    def __init__(self, stopPath=None, userDict=None):
        super(themeAnalysis, self).__init__()
        self.stopPath = stopPath
        self.userDict = userDict

    def seg(self, contents):
        """
        新闻内容分词
        params:
            contents: a list of news contents.
        """
        if self.userDict != None:
            jieba.load_userdict(self.userDict)

        stop_words = []
        if self.stopPath != None:
            stop_words = [word.strip() for word in open(self.stopPath, 'r', encoding='utf-8').readlines()]
        stop_words.append(' ')

        self.content_seg = []
        for content in contents:
            seg = [word for word in list(jieba.lcut(content)) if word not in stop_words]
            self.content_seg.append(seg) #分词结果

###话题模型
    def LSA(self, contents, num_topic=5, num_words=10):
        """
        Use LSA model to get the key words of content.
        """
        from gensim import models,corpora

        self.seg(contents)
        #self.content_seg = [' '.join(content) for content in self.content_seg] #数据
        dictionary = corpora.Dictionary(self.content_seg) #初始化词典
        corpus = [dictionary.doc2bow(content) for content in self.content_seg] #构建词袋
        tf_idf=models.TfidfModel(corpus)
        corpus_tfidf=tf_idf[corpus]
        #train the lsi model
        lsi=models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=num_topic)
        self.topics=lsi.show_topics(num_words=num_words, log=0)
        #print(self.topics)
        return self.topics

    def LDA(self, contents, num_topics=5, num_words=10):

        import gensim
        from gensim import corpora

        self.seg(contents)
        dictionary = corpora.Dictionary(self.content_seg) #初始化词典 #⌃⇧M 选择括号的内容
        doc_term_matrix = [dictionary.doc2bow(content) for content in self.content_seg]

        Lda = gensim.models.ldamodel.LdaModel
        ldamodel = Lda(doc_term_matrix, num_topics=num_topics, id2word = dictionary, passes=50)
        self.lda_topics = ldamodel.print_topics(num_topics=num_topics, num_words=num_words)
        return self.lda_topics

    def LDA_sklearn(self, contents, num_topics=5, num_words=10, max_df=0.95, min_df=2, max_features=1000, showLDA=True):

        from sklearn.decomposition import LatentDirichletAllocation
        from sklearn.feature_extraction.text import CountVectorizer

        self.seg(contents)
        tf_vectorizer = CountVectorizer(max_df=max_df, min_df=min_df, 
                            max_features=max_features, stop_words='english')

        documents = [' '.join(content) for content in self.content_seg]
        tf = tf_vectorizer.fit_transform(documents)
        feature_names = tf_vectorizer.get_feature_names()

        model = LatentDirichletAllocation(n_components=num_topics, 
                                max_iter=5, 
                                learning_method='online', 
                                learning_offset=50.,
                                random_state=0).fit(tf)

        self.LDA_sklearn_topics = []
        for topic_idx, topic in enumerate(model.components_):
            #print("Topic %d:" % (topic_idx))
            topic_info = '(%s, '%(topic_idx)
            for i in topic.argsort()[:-(num_words+1):-1]: #返回最大的index
                topic_info += str(np.around(topic[i], decimals=3))+'*'+'\"%s\"'%(feature_names[i]) + ' + '

            self.LDA_sklearn_topics.append(topic_info[:-3] + ')')
            #self.LDA_sklearn_topics.append([(feature_names[i], np.around(topic[i], decimals=3)) for i in topic.argsort()[:-(num_words+1):-1]]) #[:-a-1:-1]倒序排列，共取a个数
        #LDA可视化
        #交互图解释：一个圆圈代表一个主题，圆圈大小代表每个主题包含的文章数。
        #-->当鼠标未点到圆圈时，显示的是最重要（频率最高）的30个关键词。
        #-->当鼠标点到圆圈时，显示每个关键词在该主题下的频率。
        if showLDA == True:
            import pyLDAvis
            import pyLDAvis.sklearn

            result = pyLDAvis.sklearn.prepare(model, tf, tf_vectorizer)
            pyLDAvis.show(result)

        return self.LDA_sklearn_topics

if __name__ == '__main__':
    import pandas as pd
    analyse = themeAnalysis(stopPath='中文停用词.txt', userDict='user_dict.txt')
    df = pd.read_csv('test.csv')
    for topic in analyse.LSA(contents=df[:20].CONTENT):
        print(topic)

    print('****'*10)
    for topic in analyse.LDA(contents=df[:20].CONTENT):
        print(topic)
    print('****'*10)

    for topic in analyse.LDA_sklearn(contents=df[:20].CONTENT, showLDA=True):
        print(topic)




