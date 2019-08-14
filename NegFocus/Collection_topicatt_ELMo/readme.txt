文件说明
train.py 包含模型训练、测试以及评估函数
model.py 主模型内容
loader.py 数据集预处理
utils.py 相关模型参数初始化

数据集处理参考模板：
`` `` O punct * throwing -5 O
I PRP B-NP nsubj A0 throwing -4 I
wo MD B-VP aux AM-MOD throwing -3 O
n't RB I-VP neg AM-NEG throwing -2 O
be VB I-VP aux * throwing -1 O
throwing VBG I-VP ccomp V throwing 0 O
90 CD B-NP num A1 throwing 1 O
mph NN I-NP dobj A1 throwing 2 O
, , O punct * throwing 3 O
but CC O cc * throwing 4 O
I PRP B-NP nsubj * throwing 5 O
will MD B-VP aux * throwing 6 O
throw VB I-VP conj * throwing 7 O
80-plus JJ B-ADJP dobj * throwing 8 O
, , O punct * throwing 9 O
'' '' O punct * throwing 10 O
he PRP B-NP nsubj * throwing 11 O
says VBZ B-VP root * throwing 12 O
. . O punct * throwing 13 O
其中，第一列表示词，第二列表示词性，第三列表示语块标签，第四列表示依存节点，第五列表示语义角色，第六列表示否定动词，第七列表示相对位置，最后一列为标签（I、O）

需求
1.词向量 https://allennlp.org/elmo
2.主题模型gensim工具 https://radimrehurek.com/gensim/index.html
3.主题模型Wikipedia corpus：https://radimrehurek.com/gensim/wiki.html
3.数据集：*SEM2012评测任务 https://www.clips.uantwerpen.be/sem2012-st-neg/
4. pytorch 0.2