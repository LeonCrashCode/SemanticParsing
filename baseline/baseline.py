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

from mask import StructuredMask

use_cuda = torch.cuda.is_available()
if use_cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
    device = int(sys.argv[1])

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
        self.out = nn.Linear(self.feat_dim, self.tags_info.all_tag_size)


    def forward(self, sentence_variable, input, hidden, encoder_output, train=True, mask_variable=None):

        if train:
            self.lstm.dropout = self.dropout_p
            embedded = self.tag_embeds(input).unsqueeze(1)
            embedded = self.dropout(embedded)

            output, hidden = self.lstm(embedded, hidden)
            
            attn_weights = F.softmax(torch.bmm(output.transpose(0,1), encoder_output.transpose(0,1).transpose(1,2)).view(output.size(0),-1), 1)
            attn_hiddens = torch.bmm(attn_weights.unsqueeze(0),encoder_output.transpose(0,1))
            feat_hiddens = self.feat_tanh(self.feat(torch.cat((attn_hiddens, embedded.transpose(0,1)), 2).view(output.size(0),-1)))

            global_score = self.out(feat_hiddens)

            total_score = global_score

            output = F.log_softmax(total_score + (mask_variable - 1) * 1e10, 1)

            return output
        else:
            self.lstm.dropout = 0.0
            tokens = []
            self.mask_pool.reset()
            one_mask_variable = Variable(torch.FloatTensor([1 for i in range(decoder.tags_info.all_tag_size - decoder.tags_info.tag_size)]), volatile=True)
            zero_mask_variable = Variable(torch.FloatTensor([0 for i in range(decoder.tags_info.all_tag_size - decoder.tags_info.tag_size)]), volatile=True)
    
            while True:
                mask = self.mask_pool.get_step_mask()
                mask_variable = Variable(torch.FloatTensor(mask[:-1]), volatile=True).unsqueeze(0)
                if mask[-1] == 0:
                    mask_variable = torch.cat((mask_variable, zero_mask_variable),1)
                elif mask[-1] == 1:
                    mask_variable = torch.cat((mask_variable, one_mask_variable),1)
                mask_variable = mask_variable.cuda(device) if use_cuda else mask_variable
                embedded = self.tag_embeds(input).view(1, 1, -1)
                output, hidden = self.lstm(embedded, hidden)

                attn_weights = F.softmax(torch.bmm(output, encoder_output.transpose(0,1).transpose(1,2)).view(output.size(0), -1), 1)
                attn_hiddens = torch.bmm(attn_weights.unsqueeze(0), encoder_output.transpose(0, 1))
                feat_hiddens = self.feat_tanh(self.feat(torch.cat((attn_hiddens, embedded), 2).view(embedded.size(0),-1)))

                global_score = self.out(feat_hiddens)

                total_score = global_score

                output = total_score + (mask_variable - 1) * 1e10

                _, input = torch.max(output,1)
                idx = input.view(-1).data.tolist()[0]

                tokens.append(idx)
                self.mask_pool.update(idx)

                if idx == tags_info.tag_to_ix[tags_info.EOS]:
                    break
            return Variable(torch.LongTensor(tokens),volatile=True)


def train(sentence_variable, target_variable, gold_variable, mask, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, back_prop=True):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    sentence_length = len(sentence_variable[0])
    target_length = len(target_variable)
   
    loss = 0

    mask_variable = Variable(torch.FloatTensor(mask[0]), requires_grad=False)
    one_mask_variable = Variable(torch.FloatTensor([1 for i in range(decoder.tags_info.all_tag_size - decoder.tags_info.tag_size)]), requires_grad=False)
    zero_mask_variable = Variable(torch.FloatTensor([0 for i in range(decoder.tags_info.all_tag_size - decoder.tags_info.tag_size)]), requires_grad=False)
    decoder_input = Variable(torch.LongTensor([decoder.tags_info.tag_to_ix[SOS]]))
    if back_prop == False:
        decoder_input.volatile=True
        mask_variable.volatile=True
        one_mask_variable.volatile=True
        zero_mask_variable.volatil=True

    rest_mask_variable_list = []
    for i in range(len(mask[1])):
        if mask[1][i] == 1:
            rest_mask_variable_list.append(one_mask_variable)
        elif mask[1][i] == 0:
            rest_mask_variable_list.append(zero_mask_variable)
        else:
            assert False

    mask_variable = torch.cat((mask_variable, torch.cat(rest_mask_variable_list, 0)), 1)
    encoder_output, encoder_hidden = encoder(sentence_variable, encoder_hidden)

    mask_variable = mask_variable.cuda(device) if use_cuda else mask_variable
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

def trainIters(trn_instances, dev_instances, tst_instances, encoder, decoder, print_every=100, evaluate_every=1000, learning_rate=0.001):
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    criterion = nn.NLLLoss()

    check_point = {}
    if len(sys.argv) == 4:
        check_point = torch.load(sys.argv[3])
        encoder.load_state_dict(check_point["encoder"])
        decoder.load_state_dict(check_point["decoder"])
        if use_cuda:
            encoder = encoder.cuda(device)
            decoder = decoder.cuda(device)

    encoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()), lr=learning_rate, weight_decay=1e-4)
    decoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad, decoder.parameters()), lr=learning_rate, weight_decay=1e-4)

    if len(sys.argv) == 4:
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
    masks= []
    gold_variables = []

    for instance in trn_instances:
        #print "===========",len(mask_variables)
        decoder.mask_pool.reset()
        tmp = decoder.mask_pool.get_all_mask(instance[3])
        masks.append(([ x[:-1] for x in tmp],[ x[-1] for x in tmp]))

    for instance in trn_instances:
        sentence_variable = []
        if use_cuda:
            sentence_variable.append(Variable(instance[0]).cuda(device))
            sentence_variable.append(Variable(instance[1]).cuda(device))
            sentence_variable.append(Variable(instance[2]).cuda(device))
            target_variables.append(Variable(torch.LongTensor(instance[3])).cuda(device))
        else:
            sentence_variable.append(Variable(instance[0]))
            sentence_variable.append(Variable(instance[1]))
            sentence_variable.append(Variable(instance[2]))
            target_variables.append(Variable(torch.LongTensor(instance[3])))
        sentence_variables.append(sentence_variable)

    for instance in trn_instances:
        if use_cuda:
            gold_variables.append(Variable(torch.LongTensor(instance[3])).cuda(device))
        else:
            gold_variables.append(Variable(torch.LongTensor(instance[3])))

#==================================
    dev_sentence_variables = []
    dev_target_variables = []
    dev_masks = []
    dev_gold_variables = []

    for instance in dev_instances:
        decoder.mask_pool.reset()
        tmp = decoder.mask_pool.get_all_mask(instance[3])
        dev_masks.append(([ x[:-1] for x in tmp],[ x[-1] for x in tmp]))

    for instance in dev_instances:
        dev_sentence_variable = []
        if use_cuda:
            dev_sentence_variable.append(Variable(instance[0], volatile=True).cuda(device))
            dev_sentence_variable.append(Variable(instance[1], volatile=True).cuda(device))
            dev_sentence_variable.append(Variable(instance[2], volatile=True).cuda(device))
            dev_target_variables.append(Variable(torch.LongTensor(instance[3]), volatile=True).cuda(device))
        else:
            dev_sentence_variable.append(Variable(instance[0], volatile=True))
            dev_sentence_variable.append(Variable(instance[1], volatile=True))
            dev_sentence_variable.append(Variable(instance[2], volatile=True))
            dev_target_variables.append(Variable(torch.LongTensor(instance[3]), volatile=True))
        dev_sentence_variables.append(dev_sentence_variable)

    for instance in dev_instances:
        if use_cuda:
            dev_gold_variables.append(Variable(torch.LongTensor(instance[3]), volatile=True).cuda(device))
        else:
            dev_gold_variables.append(Variable(torch.LongTensor(instance[3]), volatile=True))

#======================================
    tst_sentence_variables = []
    tst_target_variables = []
    tst_gold_variables = []

    for instance in tst_instances:
        tst_sentence_variable = []
        if use_cuda:
            tst_sentence_variable.append(Variable(instance[0], volatile=True).cuda(device))
            tst_sentence_variable.append(Variable(instance[1], volatile=True).cuda(device))
            tst_sentence_variable.append(Variable(instance[2], volatile=True).cuda(device))
            tst_target_variables.append(Variable(torch.LongTensor(instance[3]), volatile=True).cuda(device))
        else:
            tst_sentence_variable.append(Variable(instance[0], volatile=True))
            tst_sentence_variable.append(Variable(instance[1], volatile=True))
            tst_sentence_variable.append(Variable(instance[2], volatile=True))
            tst_target_variables.append(Variable(torch.LongTensor(instance[3]), volatile=True))
        tst_sentence_variables.append(tst_sentence_variable)

    for instance in tst_instances:
        if use_cuda:
            tst_gold_variables.append(Variable(torch.LongTensor(instance[3]), volatile=True).cuda(device))
        else:
            tst_gold_variables.append(Variable(torch.LongTensor(instance[3]), volatile=True))

    idx = -1
    iter = 0
    if len(sys.argv) == 4:
        iter = check_point["iter"]
        idx = check_point["idx"]

    while True:
        if use_cuda:
            torch.cuda.empty_cache()
        idx += 1
        iter += 1
        if idx == len(trn_instances):
            idx = 0       

        loss = train(sentence_variables[idx], target_variables[idx], gold_variables[idx], masks[idx], encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
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
                dev_loss += train(dev_sentence_variables[dev_idx], dev_target_variables[dev_idx], dev_gold_variables[dev_idx], dev_masks[dev_idx], encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, back_prop=False)
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
        for tok in tokens:
            output.append(decoder.tags_info.ix_to_tag[tok])
        out.write(" ".join(output)+"\n")
        out.flush()
    out.close()
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
mask_pool = StructuredMask(tags_info)
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
    encoder = encoder.cuda(device)
    attn_decoder = attn_decoder.cuda(device)

trainIters(trn_instances, dev_instances, tst_instances, encoder, attn_decoder, print_every=1000, evaluate_every=50000, learning_rate=0.001)

