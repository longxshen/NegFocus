#!/bin/python
# encoding: utf-8
import torch
import torch.autograd as autograd
from torch.autograd import Variable
from utils import *
import math
import copy

START_TAG = '<START>'
STOP_TAG = '<STOP>'


def to_scalar(var):
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    #_, idx = torch.max(vec, 1)
    score, idx = torch.max(vec, 1)
    return to_scalar(score), to_scalar(idx)


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return Variable(tensor)


def log_sum_exp(vec):
    # vec 2D: 1 * tagset_size
    max_score = vec[0, argmax(vec)[1]]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def clones(module, N):
    'Produce N identical layers'
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
  
class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim,
                 pre_word_embeds=None, use_gpu=False,
                 use_crf=True, pos_embedding_dim=None,
		 conNode_embedding_dim=None, depNode_embedding_dim=None, semroles_embedding_dim=None, cue_embedding_dim=None, loc_embedding_dim=None, context_embedding_dim=None):
        super(BiLSTM_CRF, self).__init__()
        self.use_gpu = use_gpu
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
	self.pos_embedding_dim = pos_embedding_dim
        self.conNode_embedding_dim = conNode_embedding_dim
        self.depNode_embedding_dim = depNode_embedding_dim
        self.semroles_embedding_dim = semroles_embedding_dim
	self.cue_embedding_dim = cue_embedding_dim
	self.loc_embedding_dim = loc_embedding_dim
	self.context_embedding_dim = context_embedding_dim	

        self.n_pos = 56
        self.n_conNode = 27
        self.n_depNode = 50
        self.n_semroles = 101
	self.n_cue = vocab_size
	self.n_loc = 177

        self.use_crf = use_crf
        self.tagset_size = len(tag_to_ix)

        print('hidden_dim: %d, ' % (hidden_dim))
	
	if self.pos_embedding_dim:
            self.pos_embeds = nn.Embedding(self.n_pos, self.pos_embedding_dim)
            init_embedding(self.pos_embeds.weight)

        if self.conNode_embedding_dim:
            self.conNode_embeds = nn.Embedding(self.n_conNode, self.conNode_embedding_dim)
            init_embedding(self.conNode_embeds.weight)

        if self.depNode_embedding_dim:
            self.depNode_embeds = nn.Embedding(self.n_depNode, self.depNode_embedding_dim)
            init_embedding(self.depNode_embeds.weight)

        if self.semroles_embedding_dim:
            self.semroles_embeds = nn.Embedding(self.n_semroles, self.semroles_embedding_dim)
            init_embedding(self.semroles_embeds.weight)
	
	if self.cue_embedding_dim:
            self.cue_embeds = nn.Embedding(self.n_cue, self.cue_embedding_dim)
            init_embedding(self.cue_embeds.weight)

        if self.loc_embedding_dim:
            self.loc_embeds = nn.Embedding(self.n_loc, self.loc_embedding_dim)
            init_embedding(self.loc_embeds.weight)

	if self.context_embedding_dim:
            self.context_embeds = nn.Embedding(vocab_size, self.context_embedding_dim)
	    #init_embedding(self.context_embeds.weight)


        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        if pre_word_embeds is not None:
            self.pre_word_embeds = True
            self.word_embeds.weight = nn.Parameter(torch.FloatTensor(pre_word_embeds))
            if self.context_embedding_dim:
	        self.context_embeds.weight = nn.Parameter(torch.FloatTensor(pre_word_embeds))
        else:
            self.pre_word_embeds = False

        self.dropout = nn.Dropout(0.3)

	if True:
            self.lstm = nn.LSTM(1024*3+semroles_embedding_dim, hidden_dim, bidirectional=True)
        init_lstm(self.lstm)
        self.tanh = nn.Tanh()
        self.hidden2tag = nn.Linear(hidden_dim*2, self.tagset_size)
	self.softmax = nn.Softmax()
        self.topic2large = nn.Linear(160,300)

        init_linear(self.hidden2tag)
        init_linear(self.topic2large)

        if self.use_crf:
            self.transitions = nn.Parameter(
                torch.zeros(self.tagset_size, self.tagset_size))
            self.transitions.data[tag_to_ix[START_TAG], :] = -10000
            self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

    def _score_sentence(self, feats, tags):
        # tags is ground_truth, a list of ints, length is len(sentence)
        # feats is a 2D tensor, len(sentence) * tagset_size
        r = torch.LongTensor(range(feats.size()[0]))
        if self.use_gpu:
            r = r.cuda()
            pad_start_tags = torch.cat([torch.cuda.LongTensor([self.tag_to_ix[START_TAG]]), tags])
            pad_stop_tags = torch.cat([tags, torch.cuda.LongTensor([self.tag_to_ix[STOP_TAG]])])
        else:
            pad_start_tags = torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]), tags])
            pad_stop_tags = torch.cat([tags, torch.LongTensor([self.tag_to_ix[STOP_TAG]])])

        score = torch.sum(self.transitions[pad_stop_tags, pad_start_tags]) + torch.sum(feats[r, tags])

        return score

    def _get_lstm_features(self, sentence, pos, conNode, depNode, semroles, cue, loc, context_bef, context_aft, ELMo, ELMo_ConBef, ELMo_ConAft, Topic): 
        # LSTM main framework which contains word-attention
        #embeds = self.word_embeds(sentence)
        embeds = ELMo
	if self.pos_embedding_dim:
            pos_embedding = self.pos_embeds(pos)
        if self.conNode_embedding_dim:
            conNode_embedding = self.conNode_embeds(conNode)
        if self.depNode_embedding_dim:
            depNode_embedding = self.depNode_embeds(depNode)
        if self.semroles_embedding_dim:
            semroles_embedding = self.semroles_embeds(semroles)	
	if self.cue_embedding_dim:
            cue_embedding = self.cue_embeds(cue)
        if self.loc_embedding_dim:
            loc_embedding = self.loc_embeds(loc)
	if self.context_embedding_dim:
            context_bef_embedding = self.context_embeds(context_bef)#context_seqlen*context_embedding_dim
            context_aft_embedding = self.context_embeds(context_aft)

        Cot_bef = []
        Cot_aft = []
        is_ELMo_context_embeds = True
        if is_ELMo_context_embeds:
            context_bef_embedding = ELMo_ConBef
            context_aft_embedding = ELMo_ConAft
            if context_bef_embedding.size()[0] == 1024:
                print('The sentence has only one word!')
                context_bef_embedding = torch.unsqueeze(context_bef_embedding,0)
            if context_aft_embedding.size()[0] == 1024:
                print('The sentence has only one word!')
                context_aft_embedding = torch.unsqueeze(context_aft_embedding,0)
	    con_bef_tensor = torch.transpose(context_bef_embedding, 0, 1)#con_embedding * con_len
            con_aft_tensor = torch.transpose(context_aft_embedding, 0, 1)
              
            att_bef_tensor = torch.matmul(embeds, con_bef_tensor) / math.sqrt(context_bef_embedding.size()[1])#seq_len * con_len
            att_bef_tensor, _ = torch.max(att_bef_tensor, 1)#seq_len
	    #print(att_bef_tensor)
            # att_bef_tensor = torch.sum(att_bef_tensor, 1)#seq_len
            att_bef_tensor = torch.unsqueeze(att_bef_tensor, 0)#1 * seq_len 
            att_bef_tensor = self.softmax(att_bef_tensor)
            #print(att_bef_tensor)
            Cot_bef = list(torch.squeeze(att_bef_tensor).cpu().data)
            att_bef_tensor = torch.transpose(att_bef_tensor, 0, 1)
            att_bef_tensor = att_bef_tensor.expand(att_bef_tensor.size()[0], embeds.size()[1])#seq_len * word_dim
            att_bef_tensor = att_bef_tensor * embeds # seq_len * word_dim

            att_aft_tensor = torch.matmul(embeds, con_aft_tensor) /  math.sqrt(context_aft_embedding.size()[1])#seq_len * con_len
            att_aft_tensor, _ = torch.max(att_aft_tensor, 1)#seq_len
            # att_aft_tensor = torch.sum(att_aft_tensor, 1)#seq_len
            att_aft_tensor = torch.unsqueeze(att_aft_tensor, 0)#1 * seq_len 
            att_aft_tensor = self.softmax(att_aft_tensor)
            Cot_aft = list(torch.squeeze(att_aft_tensor).cpu().data)
            att_aft_tensor = torch.transpose(att_aft_tensor, 0 , 1)
            att_aft_tensor = att_aft_tensor.expand(att_aft_tensor.size()[0], embeds.size()[1])#seq_len * word_dim
            att_aft_tensor = att_aft_tensor * embeds # seq_len * word_dim 

	    embeds = torch.cat((att_bef_tensor,embeds), 1)#seq_len * (2*word_dim)
	    embeds = torch.cat((embeds,att_aft_tensor), 1)#seq_len * (3*word_dim)
            #print(embeds.size())

        if self.pos_embedding_dim:
            embeds = torch.cat((embeds,pos_embedding),1)
        if self.conNode_embedding_dim:
            embeds = torch.cat((embeds,conNode_embedding),1)
        if self.depNode_embedding_dim:
            embeds = torch.cat((embeds,depNode_embedding),1)
        if self.semroles_embedding_dim:
            embeds = torch.cat((embeds,semroles_embedding),1)
        if self.cue_embedding_dim:
            embeds = torch.cat((embeds,cue_embedding),1)
        if self.loc_embedding_dim:
            embeds = torch.cat((embeds, loc_embedding), 1)
        
#        Topic = self.topic2large(Topic)
#        embeds = torch.cat((embeds,Topic),1)
        embeds = embeds.unsqueeze(1)
        embeds = self.dropout(embeds)
        lstm_out, _ = self.lstm(embeds)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim*2)
        lstm_out = self.dropout(lstm_out)
        lstm_feats = self.hidden2tag(lstm_out)
         
        return lstm_feats, Cot_bef, Cot_aft

    def _forward_alg(self, feats):
        # calculate in log domain
        # feats is len(sentence) * tagset_size
        # initialize alpha with a Tensor with values all equal to -10000.
        init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.
        forward_var = autograd.Variable(init_alphas)
        if self.use_gpu:
            forward_var = forward_var.cuda()
        for feat in feats:
            emit_score = feat.view(-1, 1)
            tag_var = forward_var + self.transitions + emit_score
            max_tag_var, _ = torch.max(tag_var, dim=1)
            tag_var = tag_var - max_tag_var.view(-1, 1)
            forward_var = max_tag_var + torch.log(torch.sum(torch.exp(tag_var), dim=1)).view(1, -1) # ).view(1, -1)
        terminal_var = (forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]).view(1, -1)
        alpha = log_sum_exp(terminal_var)
        # Z(x)
        return alpha

    def viterbi_decode(self, feats):
        backpointers = []
	backpointers_score = []
        # analogous to forward
        init_vvars = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0
        forward_var = Variable(init_vvars)
        if self.use_gpu:
            forward_var = forward_var.cuda()
        for feat in feats:
            next_tag_var = forward_var.view(1, -1).expand(self.tagset_size, self.tagset_size) + self.transitions

            #_, bptrs_t = torch.max(next_tag_var, dim=1)
            bptrs_score, bptrs_t = torch.max(next_tag_var, dim=1)
	    bptrs_score = bptrs_score.squeeze().data.cpu().numpy()
            bptrs_t = bptrs_t.squeeze().data.cpu().numpy()
	    #print(bptrs_score)
            next_tag_var = next_tag_var.data.cpu().numpy()
            viterbivars_t = next_tag_var[range(len(bptrs_t)), bptrs_t]
            viterbivars_t = Variable(torch.FloatTensor(viterbivars_t))
            if self.use_gpu:
                viterbivars_t = viterbivars_t.cuda()
            forward_var = viterbivars_t + feat
            backpointers.append(bptrs_t)
	    backpointers_score.append(bptrs_score)
	    #print(backpointers_score)

        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        terminal_var.data[self.tag_to_ix[STOP_TAG]] = -10000.
        terminal_var.data[self.tag_to_ix[START_TAG]] = -10000.

        best_tag_score, best_tag_id = argmax(terminal_var.unsqueeze(0))

        path_score = terminal_var[best_tag_id]#the totoal score of the path
	best_path_score = [best_tag_score]#sequence of highest score
        best_path = [best_tag_id]#the path id of the highest score

        for bptrs_score, bptrs_t in zip(reversed(backpointers_score), reversed(backpointers)):
	    best_tag_score = bptrs_score[best_tag_id]
            best_tag_id = bptrs_t[best_tag_id]
            
	    best_path_score.append(best_tag_score)
            best_path.append(best_tag_id)
	
	best_path_score.pop()
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]
        best_path.reverse()
	best_path_score.reverse()
        return path_score, best_path, best_path_score

    def neg_log_likelihood(self, sentence, tags, pos, conNode, depNode, semroles, cue, loc, context_bef, context_aft, ELMo, ELMo_ConBef, ELMo_ConAft, Topic): 
        # train model and calculate the loss
	# sentence, tags is a list of ints
        # features is a 2D tensor, len(sentence) * self.tagset_size
        feats, _, _ = self._get_lstm_features(sentence, pos, conNode, depNode, semroles, cue, loc, context_bef, context_aft, ELMo, ELMo_ConBef, ELMo_ConAft, Topic)#Seq_len * tagset_size

        if self.use_crf:
            forward_score = self._forward_alg(feats)
            gold_score = self._score_sentence(feats, tags)
            return forward_score - gold_score
        else:
            tags = Variable(tags)
            scores = nn.functional.cross_entropy(feats, tags)
            return scores


    def forward(self, sentence, pos, conNode, depNode, semroles, cue, loc, context_bef, context_aft, ELMo, ELMo_ConBef, ELMo_ConAft, Topic): 
        # get the max score tag_seq
	feats, Cot_bef, Cot_aft = self._get_lstm_features(sentence, pos, conNode, depNode, semroles, cue, loc, context_bef, context_aft, ELMo, ELMo_ConBef, ELMo_ConAft, Topic)
  
        if self.use_crf: # viterbi to get tag_seq
            score, tag_seq, seq_score = self.viterbi_decode(feats)
            feats = feats[:, :2]
	    out_soft = torch.nn.functional.softmax(feats)    
            tag_score = [out_soft[i][tag_seq[i]].cpu().data[0] for i in range(len(tag_seq))]

        else:
            score, tag_seq = torch.max(feats, 1)
            tag_seq = list(tag_seq.cpu().data)

        return score, tag_seq, tag_score, Cot_bef, Cot_aft
