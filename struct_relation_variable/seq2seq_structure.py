    # -*- coding: utf-8 -*-
import re
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import os
import sys

from mask import OuterMask

use_cuda = torch.cuda.is_available()
if use_cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
    device = int(sys.argv[1])
    torch.cuda.manual_seed_all(12345678)

torch.manual_seed(12345678)

dev_out_dir = sys.argv[2]+"_dev/"
tst_out_dir = sys.argv[2]+"_tst/"
model_dir = sys.argv[2]+"_model/"

class EncoderRNN(nn.Module):
    def __init__(self, word_size, word_dim, pretrain_size, pretrain_dim, pretrain_embeddings, lemma_size, lemma_dim, input_dim, hidden_dim, n_layers=1, dropout_p=0.0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.word_embeds = nn.Embedding(word_size, word_dim)
        self.pretrain_embeds = nn.Embedding(pretrain_size, pretrain_dim)
        self.pretrain_embeds.weight = nn.Parameter(pretrain_embeddings, False)
        self.lemma_embeds = nn.Embedding(lemma_size, lemma_dim)
        self.dropout = nn.Dropout(self.dropout_p)

        self.embeds2input = nn.Linear(word_dim + pretrain_dim + lemma_dim, input_dim)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=self.n_layers, bidirectional=True)

    def forward(self, sentence, hidden, train=True):
        word_embedded = self.word_embeds(sentence[0])
        pretrain_embedded = self.pretrain_embeds(sentence[1])
        lemma_embedded = self.lemma_embeds(sentence[2])

        if train:
            word_embedded = self.dropout(word_embedded)
            lemma_embedded = self.dropout(lemma_embedded)
            self.lstm.dropout = self.dropout_p

        embeds = self.tanh(self.embeds2input(torch.cat((word_embedded, pretrain_embedded, lemma_embedded), 1))).view(len(sentence[0]),1,-1)
        output, hidden = self.lstm(embeds, hidden)
        return output, hidden

    def initHidden(self):
        if use_cuda:
            result = (Variable(torch.zeros(2*self.n_layers, 1, self.hidden_dim)).cuda(device),
                Variable(torch.zeros(2*self.n_layers, 1, self.hidden_dim)).cuda(device))
            return result
        else:
            result = (Variable(torch.zeros(2*self.n_layers, 1, self.hidden_dim)),
                Variable(torch.zeros(2*self.n_layers, 1, self.hidden_dim)))
            return result


class AttnDecoderRNN(nn.Module):
    def __init__(self, mask_pool, tags_info, tag_dim, input_dim, feat_dim, encoder_hidden_dim, n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.mask_pool = mask_pool
        self.tags_info = tags_info
        self.tag_size = tags_info.tag_size
        self.tag_dim = tag_dim
        self.input_dim = input_dim
        self.feat_dim = feat_dim
        self.hidden_dim = encoder_hidden_dim * 2

        self.n_layers = n_layers
        self.dropout_p = dropout_p

        self.dropout = nn.Dropout(self.dropout_p)
        self.tag_embeds = nn.Embedding(self.tags_info.all_tag_size, self.tag_dim)

        self.lstm = nn.LSTM(self.tag_dim, self.hidden_dim, num_layers= self.n_layers)

        self.feat = nn.Linear(self.hidden_dim + self.tag_dim, self.feat_dim)
        self.feat_tanh = nn.Tanh()
        self.out = nn.Linear(self.feat_dim, self.tag_size)

        self.selective_matrix = Variable(torch.randn(1, self.hidden_dim, self.hidden_dim))
        if use_cuda:
            self.selective_matrix = self.selective_matrix.cuda(device)

    def forward(self, sentence_variable, input, hidden, encoder_output, train=True, mask_variable=None):

        if train:
            self.lstm.dropout = self.dropout_p
            embedded = self.tag_embeds(input).unsqueeze(1)
            embedded = self.dropout(embedded)

            output, hidden = self.lstm(embedded, hidden)
            
            selective_score = torch.bmm(torch.bmm(output.transpose(0,1), self.selective_matrix), encoder_output.transpose(0,1).transpose(1,2)).view(output.size(0), -1)

            attn_weights = F.softmax(torch.bmm(output.transpose(0,1), encoder_output.transpose(0,1).transpose(1,2)).view(output.size(0),-1), 1)
            attn_hiddens = torch.bmm(attn_weights.unsqueeze(0),encoder_output.transpose(0,1))
            feat_hiddens = self.feat_tanh(self.feat(torch.cat((attn_hiddens, embedded.transpose(0,1)), 2).view(output.size(0),-1)))

            global_score = self.out(feat_hiddens)

            total_score = torch.cat((global_score, selective_score), 1)

            output = F.log_softmax(total_score + (mask_variable - 1) * 1e10, 1)

            return output
        else:
            self.lstm.dropout = 0.0
            tokens = []
            self.mask_pool.reset(sentence_variable[0].size(0))
            while True:
                mask = self.mask_pool.get_step_mask()
                mask_variable = Variable(torch.FloatTensor(mask), requires_grad = False, volatile=True).unsqueeze(0)
                if use_cuda:
                    mask_variable = mask_variable.cuda(device)
                embedded = self.tag_embeds(input).view(1, 1, -1)
                output, hidden = self.lstm(embedded, hidden)

                selective_score = torch.bmm(torch.bmm(output, self.selective_matrix), encoder_output.transpose(0,1).transpose(1,2)).view(output.size(0), -1)

                attn_weights = F.softmax(torch.bmm(output, encoder_output.transpose(0,1).transpose(1,2)).view(output.size(0), -1), 1)
                attn_hiddens = torch.bmm(attn_weights.unsqueeze(0), encoder_output.transpose(0, 1))
                feat_hiddens = self.feat_tanh(self.feat(torch.cat((attn_hiddens, embedded), 2).view(embedded.size(0),-1)))

                global_score = self.out(feat_hiddens)

                total_score = torch.cat((global_score, selective_score), 1)

                output = total_score + (mask_variable - 1) * 1e10

                _, input = torch.max(output,1)
                idx = input.view(-1).data.tolist()[0]

                if idx >= tags_info.tag_size:
                    type = idx - tags_info.tag_size
                    idx = sentence_variable[2][type].view(-1).data.tolist()[0]
                    tokens.append([type, idx])
                    idx += tags_info.tag_size
                    input = Variable(torch.LongTensor([idx]), volatile=True)
                    if use_cuda:
                        input = input.cuda(device)
                    self.mask_pool.update(type, idx)
                else:
                    tokens.append([-2, idx])
                    self.mask_pool.update(-2, idx)

                if idx == tags_info.tag_to_ix[tags_info.EOS]:
                    break
            return Variable(torch.LongTensor(tokens),volatile=True)


def train(sentence_variable, target_variable, gold_variable, mask_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, back_prop=True):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    sentence_length = len(sentence_variable[0])
    target_length = len(target_variable)
   
    loss = 0

    encoder_output, encoder_hidden = encoder(sentence_variable, encoder_hidden)

    decoder_input = Variable(torch.LongTensor([decoder.tags_info.tag_to_ix[SOS]]))
    if back_prop == False:
        decoder_input.volatile=True

    decoder_input = decoder_input.cuda(device) if use_cuda else decoder_input
    decoder_input = torch.cat((decoder_input, target_variable))
    
    #decoder_hidden = decoder.initHidden()
    decoder_hidden = (torch.cat((encoder_hidden[0][-2], encoder_hidden[0][-1]), 1).unsqueeze(0),torch.cat((encoder_hidden[1][-2], encoder_hidden[1][-1]), 1).unsqueeze(0))

    decoder_output = decoder(sentence_variable, decoder_input, decoder_hidden, encoder_output, train=True, mask_variable=mask_variable) 
    
    if use_cuda:
        gold_variable = torch.cat((gold_variable, Variable(torch.LongTensor([decoder.tags_info.tag_to_ix[EOS]])).cuda(device)))
    else:
        gold_variable = torch.cat((gold_variable, Variable(torch.LongTensor([decoder.tags_info.tag_to_ix[EOS]]))))

    loss += criterion(decoder_output, gold_variable)
   
    if back_prop:
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()
    
    return loss.data[0] / target_length

def decode(sentence_variable, target_variable, encoder, decoder):
    encoder_hidden = encoder.initHidden()

    encoder_output, encoder_hidden = encoder(sentence_variable, encoder_hidden)
    
    decoder_input = Variable(torch.LongTensor([decoder.tags_info.tag_to_ix[SOS]]), volatile=True)
    decoder_input = decoder_input.cuda(device) if use_cuda else decoder_input

    decoder_hidden = (torch.cat((encoder_hidden[0][-2], encoder_hidden[0][-1]), 1).unsqueeze(0),torch.cat((encoder_hidden[1][-2], encoder_hidden[1][-1]), 1).unsqueeze(0))

    tokens = decoder(sentence_variable, decoder_input, decoder_hidden, encoder_output, train=False)

    return tokens.view(-1,2).data.tolist()

######################################################################
# The whole training process looks like this:
#
# -  Start a timer
# -  Initialize optimizers and criterion
# -  Create set of training pairs
# -  Start empty losses array for plotting
#
# Then we call ``train`` many times and occasionally print the progress (%
# of examples, time so far, estimated time) and average loss.
#

def trainIters(trn_instances, dev_instances, tst_instances, encoder, decoder, print_every=100, evaluate_every=1000, learning_rate=0.001):
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    criterion = nn.NLLLoss()

    check_point = {}
    if len(sys.argv) >= 4:
        check_point = torch.load(sys.argv[3])
        encoder.load_state_dict(check_point["encoder"])
        decoder.load_state_dict(check_point["decoder"])
        if use_cuda:
            encoder = encoder.cuda(device)
            decoder = decoder.cuda(device)

    encoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()), lr=learning_rate, weight_decay=1e-4)
    decoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad, decoder.parameters()), lr=learning_rate, weight_decay=1e-4)

    if len(sys.argv) >= 4:
        encoder_optimizer.load_state_dict(check_point["encoder_optimizer"])
        decoder_optimizer.load_state_dict(check_point["decoder_optimizer"])

        for state in encoder_optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda(device)
        for state in decoder_optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda(device)


    #===============================
    sentence_variables = []
    target_variables = []
    mask_variables = []
    gold_variables = []

    for instance in trn_instances:
        decoder.mask_pool.reset(len(instance[0]))
        if use_cuda:
            mask_variables.append(Variable(torch.FloatTensor(decoder.mask_pool.get_all_mask(instance[3])), requires_grad = False).cuda(device))
        else:
            mask_variables.append(Variable(torch.FloatTensor(decoder.mask_pool.get_all_mask(instance[3])), requires_grad = False))

    for instance in trn_instances:
        sentence_variable = []
        if use_cuda:
            sentence_variable.append(Variable(instance[0]).cuda(device))
            sentence_variable.append(Variable(instance[1]).cuda(device))
            sentence_variable.append(Variable(instance[2]).cuda(device))
            target_variables.append(Variable(torch.LongTensor([ x[1] for x in instance[3]])).cuda(device))
        else:
            sentence_variable.append(Variable(instance[0]))
            sentence_variable.append(Variable(instance[1]))
            sentence_variable.append(Variable(instance[2]))
            target_variables.append(Variable(torch.LongTensor([ x[1] for x in instance[3]])))
        sentence_variables.append(sentence_variable)

    for instance in trn_instances:
        gold_list = []
        for x in instance[3]:
            if x[0] != -2:
                gold_list.append(x[0] + decoder.tags_info.tag_size)
            else:
                gold_list.append(x[1])
        if use_cuda:
            gold_variables.append(Variable(torch.LongTensor(gold_list)).cuda(device))
        else:
            gold_variables.append(Variable(torch.LongTensor(gold_list)))

#==================================
    dev_sentence_variables = []
    dev_target_variables = []
    dev_mask_variables = []
    dev_gold_variables = []

    for instance in dev_instances:
        decoder.mask_pool.reset(len(instance[0]))
        if use_cuda:
            dev_mask_variables.append(Variable(torch.FloatTensor(decoder.mask_pool.get_all_mask(instance[3])), volatile=True).cuda(device))
        else:
            dev_mask_variables.append(Variable(torch.FloatTensor(decoder.mask_pool.get_all_mask(instance[3])), volatile=True))

    for instance in dev_instances:
        dev_sentence_variable = []
        if use_cuda:
            dev_sentence_variable.append(Variable(instance[0], volatile=True).cuda(device))
            dev_sentence_variable.append(Variable(instance[1], volatile=True).cuda(device))
            dev_sentence_variable.append(Variable(instance[2], volatile=True).cuda(device))
            dev_target_variables.append(Variable(torch.LongTensor([ x[1] for x in instance[3]]), volatile=True).cuda(device))
        else:
            dev_sentence_variable.append(Variable(instance[0], volatile=True))
            dev_sentence_variable.append(Variable(instance[1], volatile=True))
            dev_sentence_variable.append(Variable(instance[2], volatile=True))
            dev_target_variables.append(Variable(torch.LongTensor([ x[1] for x in instance[3]]), volatile=True))
        dev_sentence_variables.append(dev_sentence_variable)

    for instance in dev_instances:
        dev_gold_list = []
        for x in instance[3]:
            if x[0] != -2:
                dev_gold_list.append(x[0] + decoder.tags_info.tag_size)
            else:
                dev_gold_list.append(x[1])
        if use_cuda:
            dev_gold_variables.append(Variable(torch.LongTensor(dev_gold_list), volatile=True).cuda(device))
        else:
            dev_gold_variables.append(Variable(torch.LongTensor(dev_gold_list), volatile=True))

#======================================
    tst_sentence_variables = []
    tst_target_variables = []
    tst_mask_variables = []
    tst_gold_variables = []

    for instance in tst_instances:
        decoder.mask_pool.reset(len(instance[0]))
        if use_cuda:
            tst_mask_variables.append(Variable(torch.FloatTensor(decoder.mask_pool.get_all_mask(instance[3])), volatile=True).cuda(device))
        else:
            tst_mask_variables.append(Variable(torch.FloatTensor(decoder.mask_pool.get_all_mask(instance[3])), volatile=True))

    for instance in tst_instances:
        tst_sentence_variable = []
        if use_cuda:
            tst_sentence_variable.append(Variable(instance[0], volatile=True).cuda(device))
            tst_sentence_variable.append(Variable(instance[1], volatile=True).cuda(device))
            tst_sentence_variable.append(Variable(instance[2], volatile=True).cuda(device))
            tst_target_variables.append(Variable(torch.LongTensor([ x[1] for x in instance[3]]), volatile=True).cuda(device))
        else:
            tst_sentence_variable.append(Variable(instance[0], volatile=True))
            tst_sentence_variable.append(Variable(instance[1], volatile=True))
            tst_sentence_variable.append(Variable(instance[2], volatile=True))
            tst_target_variables.append(Variable(torch.LongTensor([ x[1] for x in instance[3]]), volatile=True))
        tst_sentence_variables.append(tst_sentence_variable)

    for instance in tst_instances:
        tst_gold_list = []
        for x in instance[3]:
            if x[0] != -2:
                tst_gold_list.append(x[0] + decoder.tags_info.tag_size)
            else:
                tst_gold_list.append(x[1])
        if use_cuda:
            tst_gold_variables.append(Variable(torch.LongTensor(tst_gold_list), volatile=True).cuda(device))
        else:
            tst_gold_variables.append(Variable(torch.LongTensor(tst_gold_list), volatile=True))

    idx = -1
    iter = 0
    if len(sys.argv) >= 4:
        iter = check_point["iter"]
        idx = check_point["idx"]

    while True:
        if use_cuda:
            torch.cuda.empty_cache()
        idx += 1
        iter += 1
        if idx == len(trn_instances):
            idx = 0       

        loss = train(sentence_variables[idx], target_variables[idx], gold_variables[idx], mask_variables[idx], encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('epoch %.6f : %.10f' % (iter*1.0 / len(trn_instances), print_loss_avg))

        if iter % evaluate_every == 0:
            dev_idx = 0
            dev_loss = 0.0
            torch.save({"iter": iter, "idx":idx,  "encoder":encoder.state_dict(), "decoder":decoder.state_dict(), "encoder_optimizer": encoder_optimizer.state_dict(), "decoder_optimizer": decoder_optimizer.state_dict()}, model_dir+str(int(iter/evaluate_every))+".model")
            while dev_idx < len(dev_instances):
                if use_cuda:
                    torch.cuda.empty_cache()
                
                dev_loss += train(dev_sentence_variables[dev_idx], dev_target_variables[dev_idx], dev_gold_variables[dev_idx], dev_mask_variables[dev_idx], encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, back_prop=False)
                dev_idx += 1
            print('dev loss %.10f' % (dev_loss/len(dev_instances)))
            evaluate(dev_sentence_variables, dev_target_variables, encoder, decoder, dev_out_dir+str(int(iter/evaluate_every))+".drs")
            evaluate(tst_sentence_variables, tst_target_variables, encoder, decoder, tst_out_dir+str(int(iter/evaluate_every))+".drs")

def evaluate(sentence_variables, target_variables, encoder, decoder, path):
    out = open(path,"w")
    for idx in range(len(sentence_variables)):
        if use_cuda:
            torch.cuda.empty_cache()
        
        tokens = decode(sentence_variables[idx], target_variables[idx], encoder, decoder)

        output = []
        for type, tok in tokens:
            if type >= 0:
                output.append(decoder.tags_info.ix_to_lemma[tok])
            else:
                output.append(decoder.tags_info.ix_to_tag[tok])
        out.write(" ".join(output)+"\n")
        out.flush()
    out.close()
#####################################################################################
#####################################################################################
#####################################################################################
# main

from utils import readfile
from utils import data2instance_structure
from utils import readpretrain
from tag import Tag
#from mask import Mask

trn_file = "train.input"
dev_file = "dev.input"
tst_file = "test.input"
pretrain_file = "sskip.100.vectors"
tag_info_file = "tag.info"
#trn_file = "train.input.part"
#dev_file = "dev.input.part"
#tst_file = "test.input.part"
#pretrain_file = "sskip.100.vectors.part"
UNK = "<UNK>"

trn_data = readfile(trn_file)
word_to_ix = {UNK:0}
lemma_to_ix = {UNK:0}
ix_to_lemma = [UNK]
for sentence, _, lemmas, tags in trn_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
    for lemma in lemmas:
        if lemma not in lemma_to_ix:
            lemma_to_ix[lemma] = len(lemma_to_ix)
            ix_to_lemma.append(lemma)
#############################################
## tags
tags_info = Tag(tag_info_file, ix_to_lemma)
SOS = tags_info.SOS
EOS = tags_info.EOS
mask_pool = OuterMask(tags_info)
##############################################
##
#mask_info = Mask(tags)
#############################################
pretrain_to_ix = {UNK:0}
pretrain_embeddings = [ [0. for i in range(100)] ] # for UNK 
pretrain_data = readpretrain(pretrain_file)
for one in pretrain_data:
    pretrain_to_ix[one[0]] = len(pretrain_to_ix)
    pretrain_embeddings.append([float(a) for a in one[1:]])
print "pretrain dict size:", len(pretrain_to_ix)

dev_data = readfile(dev_file)
tst_data = readfile(tst_file)

print "word dict size: ", len(word_to_ix)
print "lemma dict size: ", len(lemma_to_ix)
print "global tag (w/o variables) dict size: ", tags_info.k_rel_start
print "global tag (w variables) dict size: ", tags_info.tag_size

WORD_EMBEDDING_DIM = 64
PRETRAIN_EMBEDDING_DIM = 100
LEMMA_EMBEDDING_DIM = 32
TAG_DIM = 128
INPUT_DIM = 100
ENCODER_HIDDEN_DIM = 256
DECODER_INPUT_DIM = 128
ATTENTION_HIDDEN_DIM = 256

encoder = EncoderRNN(len(word_to_ix), WORD_EMBEDDING_DIM, len(pretrain_to_ix), PRETRAIN_EMBEDDING_DIM, torch.FloatTensor(pretrain_embeddings), len(lemma_to_ix), LEMMA_EMBEDDING_DIM, INPUT_DIM, ENCODER_HIDDEN_DIM, n_layers=2, dropout_p=0.1)
attn_decoder = AttnDecoderRNN(mask_pool, tags_info, TAG_DIM, DECODER_INPUT_DIM, ENCODER_HIDDEN_DIM, ATTENTION_HIDDEN_DIM, n_layers=1, dropout_p=0.1)


###########################################################
# prepare training instance
trn_instances = data2instance_structure(trn_data, [(word_to_ix,0), (pretrain_to_ix,0), (lemma_to_ix,0), tags_info])
print "trn size: " + str(len(trn_instances))
###########################################################
# prepare development instance
dev_instances = data2instance_structure(dev_data, [(word_to_ix,0), (pretrain_to_ix,0), (lemma_to_ix,0), tags_info])
print "dev size: " + str(len(dev_instances))
###########################################################
# prepare test instance
tst_instances = data2instance_structure(tst_data, [(word_to_ix,0), (pretrain_to_ix,0), (lemma_to_ix,0), tags_info])
print "tst size: " + str(len(tst_instances))

print "GPU", use_cuda
if use_cuda:
    encoder = encoder.cuda(device)
    attn_decoder = attn_decoder.cuda(device)

trainIters(trn_instances, dev_instances, tst_instances, encoder, attn_decoder, print_every=1000, evaluate_every=50000, learning_rate=0.0005)

