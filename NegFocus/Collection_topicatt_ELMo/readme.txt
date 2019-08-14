File declaration
train.py  :train, test and evaulation
model.py  :main model
loader.py :dataset preprocess
utils.py  :initialize parameters


Dataset template
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

The columns denote word, part-of-speech, chunk label, dependent node, semantic role, negative word, relative position label(I,O)
