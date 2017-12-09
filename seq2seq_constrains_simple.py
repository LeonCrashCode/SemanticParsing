# -*- coding: utf-8 -*-
import re
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
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
            result = (Variable(torch.zeros(2*self.n_layers, 1, self.hidden_dim)).cuda(),
                Variable(torch.zeros(2*self.n_layers, 1, self.hidden_dim)).cuda())
            return result
        else:
            result = (Variable(torch.zeros(2*self.n_layers, 1, self.hidden_dim)),
                Variable(torch.zeros(2*self.n_layers, 1, self.hidden_dim)))
            return result


class AttnDecoderRNN(nn.Module):
    def __init__(self, tags_info, tag_dim, input_dim, feat_dim, encoder_hidden_dim, n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
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
            self.selective_matrix = self.selective_matrix.cuda()

    def forward(self, input, hidden, encoder_output, train=True):

        if train:
            self.lstm.dropout = self.dropout_p
            embedded = self.tag_embeds(input).unsqueeze(1)
            embedded = self.dropout(embedded)

            output, hidden = self.lstm(embedded, hidden)
            
            selective_score = torch.bmm(torch.bmm(output.transpose(0,1), self.selective_matrix), encoder_output.transpose(0,1).transpose(1,2)).view(output.size(0), -1)

            attn_weights = F.softmax(torch.bmm(output.transpose(0,1), encoder_output.transpose(0,1).transpose(1,2)).view(output.size(0),-1))
            attn_hiddens = torch.bmm(attn_weights.unsqueeze(0),encoder_output.transpose(0,1))
            feat_hiddens = self.feat_tanh(self.feat(torch.cat((attn_hiddens, embedded.transpose(0,1)), 2).view(output.size(0),-1)))

            global_score = self.out(feat_hiddens)

            output = F.log_softmax(torch.cat((global_score, selective_score), 1))
            return output
        else:
            self.lstm.dropout = 0.0
            tokens = []
            while True:
                embedded = self.tag_embeds(input).view(1,1,-1)
                output, hidden = self.lstm(embedded, hidden)
                attn_weights = F.softmax(torch.bmm(output.transpose(0,1), encoder_output.transpose(0,1).transpose(1,2)).view(output.size(0),-1))
                attn_hiddens = torch.bmm(attn_weights.unsqueeze(0),encoder_output.transpose(0,1))
                feat_hiddens = self.feat_tanh(self.feat(torch.cat((attn_hiddens, embedded.transpose(0,1)), 2).view(embedded.size(0),-1)))
                output = self.out(feat_hiddens).view(1,-1)
                _, input = torch.max(output,1)
                idx = input.view(-1).data.tolist()[0]
                tokens.append(idx)
                if idx == tag_to_ix[EOS] or len(tokens) == 500:
                    break
                return Variable(torch.LongTensor(tokens))
		
    def initHidden(self):
        if use_cuda:
            result = (Variable(torch.zeros(1, 1, self.hidden_dim)).cuda(),
                Variable(torch.zeros(1, 1, self.hidden_dim)).cuda())
            return result
        else:
            result = (Variable(torch.zeros(1, 1, self.hidden_dim)),
                Variable(torch.zeros(1, 1, self.hidden_dim)))
            return result

def train(sentence_variable, target_variable, gold_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    sentence_length = len(sentence_variable[0])
    target_length = len(target_variable)
   
    loss = 0

    encoder_output, encoder_hidden = encoder(sentence_variable, encoder_hidden)

    decoder_input = Variable(torch.LongTensor([decoder.tags_info.tag_to_ix[SOS]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    decoder_input = torch.cat((decoder_input, target_variable))
    
    #decoder_hidden = decoder.initHidden()
    decoder_hidden = (torch.cat((encoder_hidden[0][-2], encoder_hidden[0][-1]), 1).unsqueeze(0),torch.cat((encoder_hidden[1][-2], encoder_hidden[1][-1]), 1).unsqueeze(0))

    decoder_output = decoder(decoder_input, decoder_hidden, encoder_output, train=True) 
    

    gold_variable = torch.cat((gold_variable, Variable(torch.LongTensor([decoder.tags_info.tag_to_ix[EOS]]))))
    gold_variable = gold_variable.cuda() if use_cuda else gold_variable

    loss += criterion(decoder_output, gold_variable)
   
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.data[0] / target_length

def decode(sentence_variable, target_variable, encoder, decoder):
    encoder_hidden = encoder.initHidden()
    
    encoder_output, encoder_hidden = encoder(sentence_variable, encoder_hidden)
    
    decoder_input = Variable(torch.LongTensor([tag_to_ix[SOS]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = (torch.cat((encoder_hidden[0][-2], encoder_hidden[0][-1]), 1).unsqueeze(0),torch.cat((encoder_hidden[1][-2], encoder_hidden[1][-1]), 1).unsqueeze(0))

    tokens = decoder(decoder_input, decoder_hidden, encoder_output, train=False)

    return tokens.view(-1).data.tolist()

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

def trainIters(trn_instances, dev_instances, encoder, decoder, print_every=100, evaluate_every=1000, learning_rate=0.001):
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()), lr=learning_rate, weight_decay=1e-4)
    decoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad, decoder.parameters()), lr=learning_rate, weight_decay=1e-4)

    criterion = nn.NLLLoss()

    idx = -1
    iter = 0
    while True:
        idx += 1
        iter += 1
        if idx == len(trn_instances):
            idx = 0

        sentence_variable = []
        target_variable = Variable(torch.LongTensor([ x[1] for x in trn_instances[idx][3]]))

        gold_list = []
        for x in trn_instances[idx][3]:
            if x[0] != -2:
                gold_list.append(x[0] + decoder.tags_info.tag_size)
            else:
                gold_list.append(x[1])
        gold_variable = Variable(torch.LongTensor(gold_list))

        if use_cuda:
            sentence_variable.append(Variable(trn_instances[idx][0]).cuda())
            sentence_variable.append(Variable(trn_instances[idx][1]).cuda())
            sentence_variable.append(Variable(trn_instances[idx][2]).cuda())
            target_variable = target_variable.cuda()
        else:
            sentence_variable.append(Variable(trn_instances[idx][0]))
            sentence_variable.append(Variable(trn_instances[idx][1]))
            sentence_variable.append(Variable(trn_instances[idx][2]))        

        loss = train(sentence_variable, target_variable, gold_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('epoch %.6f : %.10f' % (iter*1.0 / len(trn_instances), print_loss_avg))

        #if iter % evaluate_every == 0:
            #evaluate(dev_instances[0], encoder, decoder)

def evaluate(instances, encoder, decoder):
    for instance in instances:
        sentence_variable = []
        target_variable = Variable(instance[3])

        if use_cuda:
            sentence_variable.append(Variable(instance[0]).cuda())
            sentence_variable.append(Variable(instance[1]).cuda())
            sentence_variable.append(Variable(instance[2]).cuda())
            target_variable = target_variable.cuda()
        else:
            sentence_variable.append(Variable(instance[0]))
            sentence_variable.append(Variable(instance[1]))
            sentence_variable.append(Variable(instance[2]))
        tokens = decode(sentence_variable, target_variable, encoder, decoder)

    for tok in tokens:
        print ix_to_tag[tok],
    print
#####################################################################################
#####################################################################################
#####################################################################################
# main

from utils import readfile
from utils import data2instance_constrains
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
PADDING = "<PADDING>"

trn_data = readfile(trn_file)
word_to_ix = {UNK:1, PADDING: 0}
lemma_to_ix = {UNK:1, PADDING: 0}
for sentence, _, lemmas, tags in trn_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
    for lemma in lemmas:
        if lemma not in lemma_to_ix:
            lemma_to_ix[lemma] = len(lemma_to_ix)
#############################################
## tags
tags_info = Tag(tag_info_file, lemma_to_ix.keys())
SOS = tags_info.SOS
EOS = tags_info.EOS

##############################################
##
#mask_info = Mask(tags)
#############################################
pretrain_to_ix = {UNK:1, PADDING: 0}
pretrain_embeddings = [ [0. for i in range(100)], [0. for i in range(100)]] # for UNK and PADDING
pretrain_data = readpretrain(pretrain_file)
for one in pretrain_data:
    pretrain_to_ix[one[0]] = len(pretrain_to_ix)
    pretrain_embeddings.append([float(a) for a in one[1:]])
print "pretrain dict size:", len(pretrain_to_ix)

dev_data = readfile(dev_file)
tst_data = readfile(tst_file)

print "word dict size: ", len(word_to_ix)
print "lemma dict size: ", len(lemma_to_ix)
print "global tag dict size: ", tags_info.tag_size

WORD_EMBEDDING_DIM = 64
PRETRAIN_EMBEDDING_DIM = 100
LEMMA_EMBEDDING_DIM = 32
TAG_DIM = 128
INPUT_DIM = 100
ENCODER_HIDDEN_DIM = 256
DECODER_INPUT_DIM = 128
ATTENTION_HIDDEN_DIM = 256

encoder = EncoderRNN(len(word_to_ix), WORD_EMBEDDING_DIM, len(pretrain_to_ix), PRETRAIN_EMBEDDING_DIM, torch.FloatTensor(pretrain_embeddings), len(lemma_to_ix), LEMMA_EMBEDDING_DIM, INPUT_DIM, ENCODER_HIDDEN_DIM, n_layers=2, dropout_p=0.1)
attn_decoder = AttnDecoderRNN(tags_info, TAG_DIM, DECODER_INPUT_DIM, ENCODER_HIDDEN_DIM, ATTENTION_HIDDEN_DIM, n_layers=1, dropout_p=0.1)


###########################################################
# prepare training instance
trn_instances = data2instance_constrains(trn_data, [(word_to_ix,0), (pretrain_to_ix,0), (lemma_to_ix,0), tags_info])
print "trn size: " + str(len(trn_instances))
###########################################################
# prepare development instance
dev_instances = data2instance_constrains(dev_data, [(word_to_ix,0), (pretrain_to_ix,0), (lemma_to_ix,0), tags_info])
print "dev size: " + str(len(dev_instances))
###########################################################
# prepare test instance
tst_instances = data2instance_constrains(tst_data, [(word_to_ix,0), (pretrain_to_ix,0), (lemma_to_ix,0), tags_info])
print "tst size: " + str(len(tst_instances))

print "GPU", use_cuda
if use_cuda:
    encoder = encoder.cuda()
    attn_decoder = attn_decoder.cuda()

trainIters(trn_instances, dev_instances, encoder, attn_decoder, print_every=1000, evaluate_every=50000, learning_rate=0.001)

