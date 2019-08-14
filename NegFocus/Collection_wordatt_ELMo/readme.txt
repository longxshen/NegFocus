File declaration
train.py  :train, test and evaulation
model.py  :main model
loader.py :dataset preprocess
utils.py  :initialize parameters


Dataset template
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

The first to seventh columns are word, part-of-speech, chunk label, dependent node, semantic role, negative word, relative position
The eighth to last but one columns are context sentences divided by ###
The last column is label(I,O)
