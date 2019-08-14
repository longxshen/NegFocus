文件说明
train.py 包含模型训练、测试以及评估函数
model.py 主模型内容
loader.py 数据集预处理
utils.py 相关模型参数初始化

数据集处理参考模板：
`` `` O punct * throwing -5 `` I 'm not so young anymore , '' concedes the cigar-chomping , 48-year-old Mr. Tiant . ### White-haired Pedro Ramos , at 54 the league 's oldest player and a pitcher-coach with the Suns , has lost even more speed . O
I PRP B-NP nsubj A0 throwing -4 `` I 'm not so young anymore , '' concedes the cigar-chomping , 48-year-old Mr. Tiant . ### White-haired Pedro Ramos , at 54 the league 's oldest player and a pitcher-coach with the Suns , has lost even more speed . I
wo MD B-VP aux AM-MOD throwing -3 `` I 'm not so young anymore , '' concedes the cigar-chomping , 48-year-old Mr. Tiant . ### White-haired Pedro Ramos , at 54 the league 's oldest player and a pitcher-coach with the Suns , has lost even more speed . O
n't RB I-VP neg AM-NEG throwing -2 `` I 'm not so young anymore , '' concedes the cigar-chomping , 48-year-old Mr. Tiant . ### White-haired Pedro Ramos , at 54 the league 's oldest player and a pitcher-coach with the Suns , has lost even more speed . O
be VB I-VP aux * throwing -1 `` I 'm not so young anymore , '' concedes the cigar-chomping , 48-year-old Mr. Tiant . ### White-haired Pedro Ramos , at 54 the league 's oldest player and a pitcher-coach with the Suns , has lost even more speed . O
throwing VBG I-VP ccomp V throwing 0 `` I 'm not so young anymore , '' concedes the cigar-chomping , 48-year-old Mr. Tiant . ### White-haired Pedro Ramos , at 54 the league 's oldest player and a pitcher-coach with the Suns , has lost even more speed . O
90 CD B-NP num A1 throwing 1 `` I 'm not so young anymore , '' concedes the cigar-chomping , 48-year-old Mr. Tiant . ### White-haired Pedro Ramos , at 54 the league 's oldest player and a pitcher-coach with the Suns , has lost even more speed . O
mph NN I-NP dobj A1 throwing 2 `` I 'm not so young anymore , '' concedes the cigar-chomping , 48-year-old Mr. Tiant . ### White-haired Pedro Ramos , at 54 the league 's oldest player and a pitcher-coach with the Suns , has lost even more speed . O
, , O punct * throwing 3 `` I 'm not so young anymore , '' concedes the cigar-chomping , 48-year-old Mr. Tiant . ### White-haired Pedro Ramos , at 54 the league 's oldest player and a pitcher-coach with the Suns , has lost even more speed . O
but CC O cc * throwing 4 `` I 'm not so young anymore , '' concedes the cigar-chomping , 48-year-old Mr. Tiant . ### White-haired Pedro Ramos , at 54 the league 's oldest player and a pitcher-coach with the Suns , has lost even more speed . O
I PRP B-NP nsubj * throwing 5 `` I 'm not so young anymore , '' concedes the cigar-chomping , 48-year-old Mr. Tiant . ### White-haired Pedro Ramos , at 54 the league 's oldest player and a pitcher-coach with the Suns , has lost even more speed . O
will MD B-VP aux * throwing 6 `` I 'm not so young anymore , '' concedes the cigar-chomping , 48-year-old Mr. Tiant . ### White-haired Pedro Ramos , at 54 the league 's oldest player and a pitcher-coach with the Suns , has lost even more speed . O
throw VB I-VP conj * throwing 7 `` I 'm not so young anymore , '' concedes the cigar-chomping , 48-year-old Mr. Tiant . ### White-haired Pedro Ramos , at 54 the league 's oldest player and a pitcher-coach with the Suns , has lost even more speed . O
80-plus JJ B-ADJP dobj * throwing 8 `` I 'm not so young anymore , '' concedes the cigar-chomping , 48-year-old Mr. Tiant . ### White-haired Pedro Ramos , at 54 the league 's oldest player and a pitcher-coach with the Suns , has lost even more speed . O
, , O punct * throwing 9 `` I 'm not so young anymore , '' concedes the cigar-chomping , 48-year-old Mr. Tiant . ### White-haired Pedro Ramos , at 54 the league 's oldest player and a pitcher-coach with the Suns , has lost even more speed . O
'' '' O punct * throwing 10 `` I 'm not so young anymore , '' concedes the cigar-chomping , 48-year-old Mr. Tiant . ### White-haired Pedro Ramos , at 54 the league 's oldest player and a pitcher-coach with the Suns , has lost even more speed . O
he PRP B-NP nsubj * throwing 11 `` I 'm not so young anymore , '' concedes the cigar-chomping , 48-year-old Mr. Tiant . ### White-haired Pedro Ramos , at 54 the league 's oldest player and a pitcher-coach with the Suns , has lost even more speed . O
says VBZ B-VP root * throwing 12 `` I 'm not so young anymore , '' concedes the cigar-chomping , 48-year-old Mr. Tiant . ### White-haired Pedro Ramos , at 54 the league 's oldest player and a pitcher-coach with the Suns , has lost even more speed . O
. . O punct * throwing 13 `` I 'm not so young anymore , '' concedes the cigar-chomping , 48-year-old Mr. Tiant . ### White-haired Pedro Ramos , at 54 the league 's oldest player and a pitcher-coach with the Suns , has lost even more speed . O
其中，第一列表示词，第二列表示词性，第三列表示语块标签，第四列表示依存节点，第五列表示语义角色，第六列表示否定动词，第七列表示相对位置，第八列到倒数第二列为上下文句子以###隔开，最后一列为标签（I、O）

需求
1.词向量 https://allennlp.org/elmo
2.主题模型gensim工具 https://radimrehurek.com/gensim/index.html
3.主题模型Wikipedia corpus：https://radimrehurek.com/gensim/wiki.html
3.数据集：*SEM2012评测任务 https://www.clips.uantwerpen.be/sem2012-st-neg/
4. pytorch 0.2