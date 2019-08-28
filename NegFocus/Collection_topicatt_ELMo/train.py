#!/bin/python
# encoding: utf-8
from __future__ import print_function
import optparse
import itertools
from collections import OrderedDict
import loader
import torch
import time
import cPickle
from torch.autograd import Variable
#import matplotlib.pyplot as plt
import sys
#import visdom
from utils import *
from loader import *
from model import BiLSTM_CRF
t = time.time()
models_path = "models/"

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

optparser = optparse.OptionParser()
optparser.add_option(
    "-T", "--train", default="dataset/SixFeature_train",
    help="Train set location"
)
optparser.add_option(
    "-d", "--dev", default="dataset/SixFeature_develop",
    help="Dev set location"
)
optparser.add_option(
    "-t", "--test", default="dataset/SixFeature_test",
    help="Test set location"
)
optparser.add_option(
    "-w", "--word_dim", default="50",
    type='int', help="Token embedding dimension"
)
optparser.add_option(
    "-W", "--word_lstm_dim", default="250",
    type='int', help="Token LSTM hidden layer size"
)
optparser.add_option(
    "-B", "--word_bidirect", default="1",
    type='int', help="Use a bidirectional LSTM for words"
)
optparser.add_option(
    "-f", "--crf", default="1",
    type='int', help="Use CRF (0 to disable)"
)
optparser.add_option(
    "-D", "--dropout", default="0.5",
    type='float', help="Droupout on the input (0 = no dropout)"
)
optparser.add_option(
    "-r", "--reload", default="0",
    type='int', help="Reload the last saved model"
)
optparser.add_option(
    "-g", '--use_gpu', default='1',
    type='int', help='whether or not to ues gpu'
)
optparser.add_option(
    '--loss', default='loss.txt',
    help='loss file location'
)
optparser.add_option(
    '--name', default='test',
    help='model name'
)

opts = optparser.parse_args()[0]

parameters = OrderedDict()
parameters['word_dim'] = opts.word_dim
parameters['word_lstm_dim'] = opts.word_lstm_dim
parameters['word_bidirect'] = opts.word_bidirect == 1
parameters['crf'] = opts.crf == 1
parameters['dropout'] = opts.dropout
parameters['reload'] = opts.reload == 1
parameters['name'] = opts.name

parameters['use_gpu'] = opts.use_gpu == 1 and torch.cuda.is_available()
use_gpu = parameters['use_gpu']

mapping_file = 'models/mapping.pkl'

name = parameters['name']
model_name = models_path + name #get_name(parameters)
tmp_model = model_name + '.tmp'


assert os.path.isfile(opts.train)
assert os.path.isfile(opts.dev)
assert os.path.isfile(opts.test)
assert 0. <= parameters['dropout'] < 1.0
#assert not parameters['pre_emb'] or parameters['word_dim'] > 0


train_sentences = loader.load_sentences(opts.train)
dev_sentences = loader.load_sentences(opts.dev)
test_sentences = loader.load_sentences(opts.test)

dico_words, word_to_id, id_to_word = word_mapping(train_sentences + test_sentences + dev_sentences)
dico_pos,pos_to_id,id_to_pos = pos_mapping(train_sentences + test_sentences + dev_sentences)
dico_conNode,conNode_to_id,id_to_conNode = conNode_mapping(train_sentences + test_sentences + dev_sentences)
dico_depNode,depNode_to_id,id_to_depNode = depNode_mapping(train_sentences + test_sentences + dev_sentences)
dico_semroles,semroles_to_id,id_to_semroles = semroles_mapping(train_sentences + test_sentences + dev_sentences)
dico_loc,loc_to_id,id_to_loc = loc_mapping(train_sentences + test_sentences + dev_sentences)
dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)

train_data = prepare_dataset(
    train_sentences, word_to_id, tag_to_id, pos_to_id, conNode_to_id, depNode_to_id, semroles_to_id, loc_to_id
)
dev_data = prepare_dataset(
    dev_sentences, word_to_id, tag_to_id, pos_to_id, conNode_to_id, depNode_to_id, semroles_to_id, loc_to_id,
)
test_data = prepare_dataset(
    test_sentences, word_to_id, tag_to_id, pos_to_id, conNode_to_id, depNode_to_id, semroles_to_id, loc_to_id
)
print("%i / %i / %i sentences in train / dev / test." % (
    len(train_data), len(dev_data), len(test_data)))


def counts(dic):
    vocab = set()
    maxlen = 0
    for w in dic:
        #w = w.lower()
        vocab.add(w)
    vocab = sorted(list(vocab))
    return vocab
vocab = counts(dico_words)
def _load_vocab(vocab_file): # load vocab，senna_words.lst
    # load vocab from file
    vocab = []
    with open(vocab_file) as f:
        for line in f:
            w = line.strip()
            vocab.append(w)
    return vocab
def _load_embedding(embed_file, words_file):
    words2id = {}
    embed = np.load(embed_file)
    words = _load_vocab(words_file)
    for id, w in enumerate(words):
        words2id[w] = id
    return embed, words2id
def final_embeddings(pretrain_embed_file, pretrain_words_file, vocab, word_to_id):
    word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(word_to_id), opts.word_dim))
    pretrain_embed, pretrain_words2id = _load_embedding(
        pretrain_embed_file,  # pretrained embedding and word
        pretrain_words_file)
    vocab_size = len(vocab)
    #print(pretrain_embed)
    # words2id = {}
    for w in vocab:
        if w in pretrain_words2id:
            idx = pretrain_words2id[w]
            word_embeds[word_to_id[w]] = pretrain_embed[idx]
        else:
            vec = np.random.normal(0, 0.1, parameters['word_dim'])
            word_embeds[word_to_id[w]] = vec
            
    #word_embeds['<UNK>'] = np.zeros(parameters['word_dim'])
    #word_embed[0] = np.zeros(parameters['word_dim'])
    return word_embeds, pretrain_words2id

pretrain_embed_file = "models/embed50.senna.npy"
pretrain_words_file = "models/senna_words.lst"
word_embeds, pretrain_words2id = final_embeddings(pretrain_embed_file, pretrain_words_file, vocab, word_to_id)
print('Loaded %i pretrained embeddings.' % len(pretrain_words2id))


fELMo_traddde = "models/ELMo_tr&de_vector.npy"
fELMo_te = "models/ELMo_te_vector.npy"
emlo_traddde = np.load(fELMo_traddde)
emlo_te = np.load(fELMo_te)

ftopic_traddde = "models/Large1_LsiTopic80Com_tr&de.npy"
ftopic_te = "models/Large1_LsiTopic80Com_te.npy"
embed_traddde = np.load(ftopic_traddde)
embed_te = np.load(ftopic_te)

with open(mapping_file, 'wb') as f:
    mappings = {
        'word_to_id': word_to_id,
        'tag_to_id': tag_to_id,
        'parameters': parameters,
        'word_embeds': word_embeds
    }
    cPickle.dump(mappings, f)

print('word_to_id: ', len(word_to_id))
model = BiLSTM_CRF(vocab_size=len(word_to_id),
                   tag_to_ix=tag_to_id,
                   embedding_dim=parameters['word_dim'],
                   hidden_dim=parameters['word_lstm_dim'],
                   use_gpu=use_gpu,
                   pre_word_embeds=word_embeds,
                   use_crf=True,     #parameters['crf'],
		   #pos_embedding_dim=10,
                   #conNode_embedding_dim=1000,
                   #depNode_embedding_dim=10,
                   semroles_embedding_dim=2000,
		   #cue_embedding_dim=10,
		   loc_embedding_dim=None)

if parameters['reload']:
    model.load_state_dict(torch.load(model_name))
if use_gpu:
    model.cuda()
learning_rate = 0.015

parameters_alg = itertools.ifilter(lambda p: p.requires_grad, model.parameters())

optimizer = torch.optim.SGD(parameters_alg, lr=learning_rate, momentum=0.9)


losses = []
loss = 0.0
best_dev_F = -1.0
best_test_F = -1.0
best_train_F = -1.0

best_dev_Acc = -1.0
best_test_Acc = -1.0
best_train_Acc = -1.0
best_dev_Acc_post = -1.0
best_test_Acc_post = -1.0
best_train_Acc_post = -1.0
max_epoch = 0
max_epoch_post = 0
senchangeby_post = ''

all_F = [[0, 0, 0]]
plot_every = 10
eval_every = 20
count = 0
#vis = visdom.Visdom()
sys.stdout.flush()

def Score_eval(prediction, datas, isTest, list_SemRole, Score_Seq, prelabel_seq):# evaluation function，calculate the accuracy
    global best_test_Acc_post
    sentence_index = 1
    if isTest:
        l_f = []
        l_f_post = []
        l_f_error = []
        l_f_post_error = []
	
    sentence_word = []
    senGold_Lab = [] # golden label
    senPre_Lab = [] # predicted label
    senPostPre_Lab = [] # predicted label after post-processing

    ifAcc = 1
    ifAcc_post = 1

    acc_num_post = 0.0
    acc_num = 0.0
    ans_AT = prelabel_seq
    SemRoles_test = list_SemRole
    score = Score_Seq   


    # -----------------------------------------------------------post-processing---------------------------------------------------------------#
    count_update_label = 0 
    count_update_label1 = 0
    count_sen = 0 
    tip = 0 
    PCS_index = 0
    SemRoles_index = 0
    temp_SemRoles = ''
    max_score = 0.0
    max_SemRoles = ''
    list_maxSemRoles = []
    last_senfinindex = 0
    while True:
        if SemRoles_index >= len(ans_AT):
            break
        if ans_AT[SemRoles_index].__contains__('I'):
            current_SemRoles = SemRoles_test[SemRoles_index].strip('\r') 
            current_score = score[SemRoles_index] 
            
            if current_score > max_score and current_SemRoles.__contains__('*') == False and current_SemRoles != '-' and current_SemRoles != 'V':
                max_score = current_score
                max_SemRoles = current_SemRoles
        if ans_AT[SemRoles_index] == '':
            if max_SemRoles == '' and max_score == 0.0: 
                max_score = 10000.0
                i2 = SemRoles_index - 1
                if PCS_index - 1 >= 0:
                    i1 = last_senfinindex + 2
                else:
                    i1 = 0
                ii = i1
    
                while ii <= i2:
                    current_SemRoles = SemRoles_test[ii].strip('\r') 
                    current_score = score[ii] 
                    if current_SemRoles.__contains__('*') or current_SemRoles == '-' or current_SemRoles == 'V':
                        ii += 1
                        continue
                    if current_score < max_score: 
                        max_score = current_score
                        max_SemRoles = current_SemRoles
                    ii += 1
            PCS_index += 1
            list_maxSemRoles.append(max_SemRoles) 
            max_SemRoles = ''
            max_score = 0.0
            last_senfinindex = SemRoles_index - 1
        SemRoles_index += 1 

    PCS_index = 0
    SemRoles_index = 0
    temp_SemRoles = list_maxSemRoles[0]
    while True:  # post-processing based on semantic roles------------------------------------------------------------------------------
        if SemRoles_index >= len(ans_AT):
            break
        label = ans_AT[SemRoles_index]
        current_SemRoles = SemRoles_test[SemRoles_index].strip('\r') 
        if current_SemRoles.__contains__('*') or current_SemRoles == '-' or current_SemRoles == 'V':  
            if ans_AT[SemRoles_index] == 'I':  # .__contains__('I'):
                count_update_label += 1
                if tip == 0:
                    count_sen += 1
                    tip = 1
            ans_AT[SemRoles_index] = 'O'
            pass
        elif current_SemRoles != temp_SemRoles and temp_SemRoles != '' and label != '': 
            if ans_AT[SemRoles_index] == 'I':  # .__contains__('I'):
                count_update_label += 1
                if tip == 0:
                    count_sen += 1
                    tip = 1
                ans_AT[SemRoles_index] = 'O'
        elif current_SemRoles == temp_SemRoles and temp_SemRoles != '' and label != '': 
            if ans_AT[SemRoles_index] == 'O':  # .__contains__('O'):
                count_update_label1 += 1
                if tip == 0:
                    count_sen += 1
                    tip = 1
                ans_AT[SemRoles_index] = 'I'
        if ans_AT[SemRoles_index] == '':
            PCS_index += 1
            if SemRoles_index + 1 >= len(ans_AT):
                break
            temp_SemRoles = list_maxSemRoles[PCS_index]
            tip = 0
        SemRoles_index += 1 
    print('后处理法更改实例的个数为--------------------：' + str(count_update_label) + '#' + str(count_update_label1) + ' 对应句子数：' + str(count_sen))
    str_change = '后处理法更改实例的个数为--------------------：' + str(count_update_label) + '#' + str(count_update_label1) + ' 对应句子数：' + str(count_sen)

    # ------------------------------------------------------------post-processing----------------------------------------------------------------------------------------#
    
    cur_sem_id = -1
    last_sem_id = 0
    for e, ans in zip(prediction, ans_AT):
        cur_sem_id += 1
        if e == '':
            if ifAcc == 1:
               acc_num += 1
            if ifAcc_post == 1:
                acc_num_post += 1

            if isTest:
                for W, GL, PPL, PL, semrole, sco in zip(sentence_word, senGold_Lab, senPostPre_Lab, senPre_Lab, SemRoles_test[last_sem_id:cur_sem_id], score[last_sem_id:cur_sem_id]):
                    l_f.append(' GL: '+GL+'  PL: '+PL+'  semrole: '+semrole+'  Sen: '+str(sentence_index)+'  W: '+W+'  Score: '+str(sco)+'\n')
                    l_f_post.append(' GL: '+GL+'  PPL: '+PPL+'  semrole: '+semrole+'  Sen: '+str(sentence_index)+'  W: '+W+'  Score: '+str(sco)+'\n')
                l_f.append('\n')
                l_f_post.append('\n')
                if ifAcc == 0:
                    for W, GL, PPL, PL, semrole, sco in zip(sentence_word, senGold_Lab, senPostPre_Lab, senPre_Lab, SemRoles_test[last_sem_id:cur_sem_id], score[last_sem_id:cur_sem_id]):
                        l_f_error.append(' GL: '+GL+'  PL: '+PL+'  semrole: '+semrole+'  Sen: '+str(sentence_index)+'  W: '+W+'  Score: '+str(sco)+'\n')
                    l_f_error.append('\n')
                if ifAcc_post == 0:
                    for W, GL, PPL, PL, semrole, sco in zip(sentence_word, senGold_Lab, senPostPre_Lab, senPre_Lab, SemRoles_test[last_sem_id:cur_sem_id], score[last_sem_id:cur_sem_id]):
                        l_f_post_error.append(' GL: '+GL+'  PPL: '+PPL+'  semrole: '+semrole+'  Sen: '+str(sentence_index)+'  W: '+W+'  Score: '+str(sco)+'\n')
                    l_f_post_error.append('\n')

            sentence_index += 1
            sentence_word = []
            senGold_Lab = []
            senPre_Lab = []
            senPostPre_Lab = []
            ifAcc = 1
            ifAcc_post = 1
            last_sem_id = cur_sem_id+1
        else:
            e = e.split()
            word = e[0]#词
            gold_label = e[1]
            pred_label = e[2]
            post_pred_label = ans
            sentence_word.append(word)
            senGold_Lab.append(gold_label)
            senPre_Lab.append(pred_label)
            senPostPre_Lab.append(post_pred_label)
            #if gold_label != pred_label:
            #    ifAcc = 0
	    if gold_label != post_pred_label:
                ifAcc_post = 0
            if gold_label != pred_label:
                ifAcc = 0

    Acc = acc_num/len(datas)
    Acc_post = acc_num_post/len(datas)
    print(len(datas))
    if isTest and Acc_post > best_test_Acc_post:
        f = codecs.open("evaluation/ans_label", 'w', 'utf-8')
        f_post = codecs.open("evaluation/ansPost_label", 'w', 'utf-8')
        f_error = codecs.open("evaluation/err_label", 'w', 'utf-8')
        f_post_error = codecs.open("evaluation/errPost_label", 'w', 'utf-8')
        f.writelines(l_f)
        f_post.writelines(l_f_post)
        f_error.writelines(l_f_error)
        f_post_error.writelines(l_f_post_error)

        f.close()
        f_post.close()
        f_error.close()
        f_post_error.close()
    return Acc, Acc_post, str_change


def evaluating(model, datas, best_Acc, best_Acc_post, is_Test, epoch): # evaluation function
    global max_epoch
    global max_epoch_post
    global senchangeby_post

    prediction = []
    save = False
    new_F = 0.0
    confusion_matrix = torch.zeros((len(tag_to_id)-2, len(tag_to_id)-2))
    list_SemRole = []
    Score_Seq = []
    prelabel_seq = []
    index = -1

    for data in datas:
        index += 1
        ground_truth_id = data['tags']
        words = data['str_words']
        pos = data['pos']
        conNode = data['constituency_node']
        depNode = data['dependency_node']
        semroles = data['semroles']
        semroles_word = data['semroles_word']
        cue = data['cue']
        loc = data['loc']

        list_SemRole.extend(semroles_word)
        list_SemRole.append('')

        dwords = Variable(torch.LongTensor(data['words']))
        dpos = Variable(torch.LongTensor(pos))
        dconNode = Variable(torch.LongTensor(conNode))
        ddepNode = Variable(torch.LongTensor(depNode))
        dsemroles = Variable(torch.LongTensor(semroles))
        dcue = Variable(torch.LongTensor(cue))
        dloc = Variable(torch.LongTensor(loc))
        #Topic = Variable(torch.FloatTensor(embed_te[index][:, :80]))
        Topic = Variable(torch.FloatTensor(embed_te[index]))
        ELMo = Variable(torch.FloatTensor(emlo_te[index]))

        if use_gpu:
            val, out, tag_score = model(dwords.cuda(), dpos.cuda(), dconNode.cuda(), ddepNode.cuda(), dsemroles.cuda(), dcue.cuda(), dloc.cuda(), Topic.cuda(), ELMo.cuda())#score , tag_sequence
        else:
            val, out, tag_score = model(dwords, dpos, dconNode, ddepNode, dsemroles, dcue, dloc, Topic, ELMo)
        predicted_id = out
        Score_Seq.extend(tag_score)
        Score_Seq.append('')
	
        for (word, true_id, pred_id) in zip(words, ground_truth_id, predicted_id):
            line = ' '.join([word, id_to_tag[true_id], id_to_tag[pred_id]])
            prediction.append(line)
            prelabel_seq.append(id_to_tag[pred_id])
            confusion_matrix[true_id, pred_id] += 1
        prediction.append('')
        prelabel_seq.append('')

    Acc, Acc_post, str_change = Score_eval(prediction,datas, is_Test, list_SemRole, Score_Seq, prelabel_seq)
    new_Acc = Acc
    new_Acc_post = Acc_post
    if new_Acc > best_Acc:
        best_Acc = new_Acc
        #save = True
	if is_Test:
            max_epoch = epoch
    if new_Acc_post > best_Acc_post:
        best_Acc_post = new_Acc_post
        senchangeby_post = str_change
        save = True
        if is_Test:
            max_epoch_post = epoch

    return best_Acc, new_Acc, best_Acc_post, new_Acc_post, save

# main function
model.train(True)
for epoch in range(1, 51):
    #for i, index in enumerate(np.random.permutation(len(train_data))):
    for i, index in enumerate(np.random.permutation(len(train_data+dev_data))):
        tr = time.time()
        count += 1
        #data = train_data[index]
        mix_data = train_data + dev_data
        data = mix_data[index]
        model.zero_grad()

        sentence_in = data['words']
        sentence_in = Variable(torch.LongTensor(sentence_in))
        tags = data['tags']
        pos = data['pos']
        conNode = data['constituency_node']
        depNode = data['dependency_node']
        semroles = data['semroles']
        cue = data['cue']
        loc = data['loc']

        targets = torch.LongTensor(tags)
        pos = Variable(torch.LongTensor(pos))
        conNode = Variable(torch.LongTensor(conNode))
        depNode = Variable(torch.LongTensor(depNode))
        semroles = Variable(torch.LongTensor(semroles))
        cue = Variable(torch.LongTensor(cue))
        loc = Variable(torch.LongTensor(loc))
        ELMo = Variable(torch.FloatTensor(emlo_traddde[index]))
        #print(emlo_traddde[index].shape)
        #Topic = Variable(torch.FloatTensor(embed_traddde[index][:, :80]))
        Topic = Variable(torch.FloatTensor(embed_traddde[index]))
        #print(embed_traddde[index].shape)

        if use_gpu:
            neg_log_likelihood = model.neg_log_likelihood(sentence_in.cuda(), targets.cuda(),
							  pos.cuda(), conNode.cuda(), depNode.cuda(), semroles.cuda(), cue.cuda(), loc.cuda(), Topic.cuda(), ELMo.cuda())
        else:
            neg_log_likelihood = model.neg_log_likelihood(sentence_in, targets,
							  pos, conNode, depNode, semroles, cue, loc, Topic, ELMo)

        loss += neg_log_likelihood.data[0] / len(data['words'])
        neg_log_likelihood.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
        optimizer.step()

        if count % plot_every == 0:
            loss /= plot_every
            print(count, ': ', loss)
            if losses == []:
                losses.append(loss)
            losses.append(loss)
            loss = 0.0

        if count % 50 == 0:
            model.train(False)
            print('epoch:---- '+str(epoch))
            best_test_Acc, new_test_Acc, best_test_Acc_post, new_test_Acc_post, _ = evaluating(model, test_data, best_test_Acc, best_test_Acc_post, True, epoch)
            print('test Acc: ' + str(new_test_Acc))
            print('test Acc postprocessing: ' + str(new_test_Acc_post))

            if count % 500 == 0:
                print('epoch: ' + str(max_epoch) + '  Best test Acc: ' + str(best_test_Acc))
                print('epoch: ' + str(max_epoch_post) + '  Best test Acc postprocessing: ' + str(best_test_Acc_post))
                print(senchangeby_post)
            sys.stdout.flush()

            model.train(True)

        if count % len(mix_data) == 0:
            adjust_learning_rate(optimizer, lr=learning_rate/(1+0.05*count/len(mix_data)))


print('epoch: ' + str(max_epoch) + '  All Best test Acc: ' + str(best_test_Acc))
print('epoch: ' + str(max_epoch_post) + '  All Best test Acc postprocessing: ' + str(best_test_Acc_post))
delta = time.time() - t
print('hour:%d , minute:%d , second:%d' % (delta/3600,delta%3600/60,delta%3600%60))
fp = open("result.txt",'w')
fp.write('epoch: ' + str(max_epoch) + '  All Best test Acc: ' + str(best_test_Acc))
fp.write('epoch: ' + str(max_epoch) + '  All Best test Acc postprocessing: ' + str(best_test_Acc_post))

fp.close()

