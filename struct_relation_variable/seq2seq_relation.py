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
import types

from mask import RelationMask

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

class SimpleRNN(nn.Module):
    def __init__(self, structure_size, tag_dim, hidden_dim, n_layers=1, dropout_p=0.0):
        super(SimpleRNN, self).__init__()
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.tag_dim = tag_dim
        self.hidden_dim = hidden_dim

        self.structure_embeds = nn.Embedding(structure_size, self.tag_dim)
        self.dropout = nn.Dropout(self.dropout_p)

        self.lstm = nn.LSTM(tag_dim, hidden_dim, num_layers=self.n_layers, bidirectional=True)

    def forward(self, sentence, hidden, train=True):

        structure_embedded = self.structure_embeds(sentence)
        if train:
            structure_embedded = self.dropout(structure_embedded)
            self.lstm.dropout = self.dropout_p

        output, hidden = self.lstm(structure_embedded.unsqueeze(1), hidden)
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
        self.total_rel = 0

        self.n_layers = n_layers
        self.dropout_p = dropout_p

        self.dropout = nn.Dropout(self.dropout_p)
        self.tag_embeds = nn.Embedding(self.tags_info.all_tag_size, self.tag_dim)

        self.condition2input = nn.Linear(self.hidden_dim, self.tag_dim)

        self.lstm = nn.LSTM(self.tag_dim, self.hidden_dim, num_layers= self.n_layers)

        self.feat = nn.Linear(self.hidden_dim + self.tag_dim, self.feat_dim)
        self.feat_tanh = nn.Tanh()
        self.out = nn.Linear(self.feat_dim, self.tag_size)

        self.selective_matrix = Variable(torch.randn(1, self.hidden_dim, self.hidden_dim))
        if use_cuda:
            self.selective_matrix = self.selective_matrix.cuda(device)

    def forward(self, sentence_variable, inputs, mask_variable, hidden, encoder_output, least, train=True):
        if train:
            self.lstm.dropout = self.dropout_p
            List = []
            for condition, input in inputs:
                List.append(self.condition2input(condition).view(1, 1, -1))
                if type(input) == types.NoneType:
                    pass
                else:
                    List.append(self.tag_embeds(input).unsqueeze(1))
            embedded = torch.cat(List, 0)
            embedded = self.dropout(embedded)

            output, hidden = self.lstm(embedded, hidden)
            selective_score = torch.bmm(torch.bmm(output.transpose(0,1), self.selective_matrix), encoder_output.transpose(0,1).transpose(1,2)).view(output.size(0), -1)

            attn_weights = F.softmax(torch.bmm(output.transpose(0,1), encoder_output.transpose(0,1).transpose(1,2)).view(output.size(0),-1), 1)
            attn_hiddens = torch.bmm(attn_weights.unsqueeze(0),encoder_output.transpose(0,1))
            feat_hiddens = self.feat_tanh(self.feat(torch.cat((attn_hiddens, embedded.transpose(0,1)), 2).view(output.size(0),-1)))

            global_score = self.out(feat_hiddens)

            total_score = torch.cat((global_score, selective_score), 1)

            output = F.log_softmax(total_score + (mask_variable - 1) *1e10, 1)

            return output
        else:
            self.lstm.dropout = 0.0
            tokens = []
            rel = 0

            mask_variable_true = Variable(torch.FloatTensor(mask_pool.get_one_mask(True)), requires_grad = False)
            mask_variable_false = Variable(torch.FloatTensor(mask_pool.get_one_mask(False)), requires_grad = False)
            if use_cuda:
                mask_variable_true = mask_variable_true.cuda(device)
                mask_variable_false = mask_variable_false.cuda(device)

            embedded = self.condition2input(inputs).view(1, 1,-1)
            while True:
                
                output, hidden = self.lstm(embedded, hidden)

                selective_score = torch.bmm(torch.bmm(output, self.selective_matrix), encoder_output.transpose(0,1).transpose(1,2)).view(output.size(0), -1)

                attn_weights = F.softmax(torch.bmm(output, encoder_output.transpose(0,1).transpose(1,2)).view(output.size(0), -1), 1)
                attn_hiddens = torch.bmm(attn_weights.unsqueeze(0), encoder_output.transpose(0, 1))
                feat_hiddens = self.feat_tanh(self.feat(torch.cat((attn_hiddens, embedded), 2).view(embedded.size(0),-1)))

                global_score = self.out(feat_hiddens)

                total_score = torch.cat((global_score, selective_score), 1)

                if least:
                    output = total_score + (mask_variable_true - 1) * 1e10
                    least = False
                else:
                    output = total_score + (mask_variable_false - 1) * 1e10

                _, input = torch.max(output,1)
                idx = input.view(-1).data.tolist()[0]

                if idx >= tags_info.tag_size:
                    ttype = idx - tags_info.tag_size
                    idx = sentence_variable[2][ttype].view(-1).data.tolist()[0]
                    tokens.append([ttype, idx])
                    idx += tags_info.tag_size
                    input = Variable(torch.LongTensor([idx]), volatile=True)
                    if use_cuda:
                        input = input.cuda(device)
                else:
                    tokens.append([-2, idx])

                if idx == tags_info.tag_to_ix[tags_info.EOS]:
                    break
                elif rel > 61 or self.total_rel > 121:
                    embedded = self.tag_embeds(input).view(1, 1, -1)
                    output, hidden = self.lstm(embedded, hidden)
                    break
                rel += 1
                self.total_rel += 1
                embedded = self.tag_embeds(input).view(1, 1, -1)
            return Variable(torch.LongTensor(tokens), volatile=True), hidden


def train(sentence_variable, struct_variable, target_variables, gold_variables, mask_variables, encoder, s_encoder, decoder, encoder_optimizer, s_encoder_optimizer, decoder_optimizer, criterion, back_prop=True):
    encoder_hidden = encoder.initHidden()
    s_encoder_hidden = s_encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    s_encoder_optimizer.zero_grad()

    target_length = 0
    loss = 0

    encoder_output, encoder_hidden = encoder(sentence_variable, encoder_hidden)
    s_encoder_output, s_encoder_hidden = s_encoder(struct_variable, s_encoder_hidden)

    decoder_hidden = (torch.cat((encoder_hidden[0][-2], encoder_hidden[0][-1]), 1).unsqueeze(0),torch.cat((encoder_hidden[1][-2], encoder_hidden[1][-1]), 1).unsqueeze(0))
    decoder_input = []
    structs = struct_variable.view(-1).data.tolist()
    p = 0
    total_rel = 0
    for i in range(len(structs)):
        if structs[i] == 5 or structs[i] == 6:
            decoder_input.append((s_encoder_output[i],target_variables[p]))
            p += 1
    decoder.total_rel = 0
    decoder_output = decoder(sentence_variable, decoder_input, mask_variables, decoder_hidden, encoder_output, least=None, train=True)

    loss += criterion(decoder_output, gold_variables)
    target_length += gold_variables.size(0)
    assert p == len(target_variables)

    if back_prop:
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        s_encoder_optimizer.step()
    
    return loss.data[0] / target_length

def decode(sentence_variable, struct_variable, encoder, s_encoder, decoder):
    encoder_hidden = encoder.initHidden()
    s_encoder_hidden = s_encoder.initHidden()

    encoder_output, encoder_hidden = encoder(sentence_variable, encoder_hidden)
    s_encoder_output, s_encoder_hidden = s_encoder(struct_variable, s_encoder_hidden)
    
    decoder_hidden = (torch.cat((encoder_hidden[0][-2], encoder_hidden[0][-1]), 1).unsqueeze(0),torch.cat((encoder_hidden[1][-2], encoder_hidden[1][-1]), 1).unsqueeze(0))

    decoder.mask_pool.reset(sentence_variable[0].size(0))
    structs = struct_variable.view(-1).data.tolist()

    all_tokens = []
    decoder.total_rel = 0
    for i in range(len(structs)):
        if structs[i] == 5 or structs[i] == 6:
            least = False
            if structs[i] == 5 or (structs[i] == 6 and structs[i+1] == 4):
                least = True
            decoder.mask_pool.set_sdrs(structs[i] == 5)
            tokens, decoder_hidden = decoder(sentence_variable, s_encoder_output[i], None, decoder_hidden, encoder_output, least, train=False)

            all_tokens.append(tokens.view(-1,2).data.tolist())
    return all_tokens

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

def trainIters(trn_instances, dev_instances, tst_instances, dev_struct_instances, tst_struct_instances, encoder, s_encoder, decoder, print_every=100, evaluate_every=1000, learning_rate=0.001):
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    criterion = nn.NLLLoss()

    check_point = {}
    if len(sys.argv) == 4:
        check_point = torch.load(sys.argv[3])
        encoder.load_state_dict(check_point["encoder"])
        decoder.load_state_dict(check_point["decoder"])
        s_encoder.load_state_dict(check_point["s_encoder"])
        if use_cuda:
            encoder = encoder.cuda(device)
            decoder = decoder.cuda(device)
            s_encoder = s_encoder.cuda(device)

    encoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()), lr=learning_rate, weight_decay=1e-4)
    decoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad, decoder.parameters()), lr=learning_rate, weight_decay=1e-4)
    s_encoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad, s_encoder.parameters()), lr=learning_rate, weight_decay=1e-4)

    if len(sys.argv) == 4:
        encoder_optimizer.load_state_dict(check_point["encoder_optimizer"])
        decoder_optimizer.load_state_dict(check_point["decoder_optimizer"])
        s_encoder_optimizer.load_state_dict(check_point["s_encoder_optimizer"])

        for state in encoder_optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda(device)
        for state in decoder_optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda(device)
        for state in s_encoder_optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda(device)


    #===============================
    sentence_variables = []
    target_variables = []
    gold_variables = []
    mask_variables = []

    for instance in trn_instances:
        sentence_variables.append([])
        if use_cuda:
            sentence_variables[-1].append(Variable(instance[0]).cuda(device))
            sentence_variables[-1].append(Variable(instance[1]).cuda(device))
            sentence_variables[-1].append(Variable(instance[2]).cuda(device))
	    sentence_variables[-1].append(Variable(torch.LongTensor([x[1] for x in instance[3]])).cuda(device))
        else:
            sentence_variables[-1].append(Variable(instance[0]))
            sentence_variables[-1].append(Variable(instance[1]))
            sentence_variables[-1].append(Variable(instance[2]))
            sentence_variables[-1].append(Variable(torch.LongTensor([x[1] for x in instance[3]])))

        target_variable = []
        all_relations = []
        p = 0
        for i in range(len(instance[3])):
            type, idx = instance[3][i]
            if idx == 5 or idx == 6:
                all_relations = all_relations + instance[4][p]
                all_relations.append([-2, 1])
                if len(instance[4][p]) == 0:
                    target_variable.append(None)
                elif use_cuda:
                    target_variable.append(Variable(torch.LongTensor([x[1] for x in instance[4][p]])).cuda(device))
                else:
                    target_variable.append(Variable(torch.LongTensor([x[1] for x in instance[4][p]])))
                p += 1
        target_variables.append(target_variable)

        sel_gen_relations = []
        for type, idx in all_relations:
            if type == -2:
                sel_gen_relations.append(idx)
            else:
                sel_gen_relations.append(type+decoder.tags_info.tag_size)
        if use_cuda:
            gold_variables.append(Variable(torch.LongTensor(sel_gen_relations)).cuda(device))
        else:
            gold_variables.append(Variable(torch.LongTensor(sel_gen_relations)))
        assert p == len(instance[4])

        decoder.mask_pool.reset(len(instance[0]))
        mask = []
        p = 0
        for i in range(len(instance[3])):
            type, idx = instance[3][i]
            if idx == 5 or idx == 6:
                least = False
                if idx == 5 or (idx == 6 and instance[3][i+1][1] == 4):
                    least = True
                decoder.mask_pool.set_sdrs(idx == 5)
                temp_mask = decoder.mask_pool.get_all_mask(len(instance[4][p]), least)
                for k in range(len(instance[4][p])):
                    temp_idx = instance[4][p][k][0]
                    if temp_idx == -2:
                        temp_idx = instance[4][p][k][1]
                    else:
                        temp_idx += decoder.tags_info.tag_size
                    assert temp_mask[k][temp_idx] == decoder.mask_pool.need
                mask = mask + temp_mask
                p += 1
        assert p == len(instance[4])

        if use_cuda:
            mask_variables.append(Variable(torch.FloatTensor(mask), requires_grad=False).cuda(device))
        else:
            mask_variables.append(Variable(torch.FloatTensor(mask), requires_grad=False))
#==================================
    dev_sentence_variables = []
    dev_target_variables = []
    dev_gold_variables = []
    dev_mask_variables = []
    dev_pred_struct_variables = []

    for instance in dev_instances:
        dev_sentence_variables.append([])
        if use_cuda:
            dev_sentence_variables[-1].append(Variable(instance[0], volatile=True).cuda(device))
            dev_sentence_variables[-1].append(Variable(instance[1], volatile=True).cuda(device))
            dev_sentence_variables[-1].append(Variable(instance[2], volatile=True).cuda(device))
            dev_sentence_variables[-1].append(Variable(torch.LongTensor([x[1] for x in instance[3]]), volatile=True).cuda(device))
        else:
            dev_sentence_variables[-1].append(Variable(instance[0], volatile=True))
            dev_sentence_variables[-1].append(Variable(instance[1], volatile=True))
            dev_sentence_variables[-1].append(Variable(instance[2], volatile=True))
            dev_sentence_variables[-1].append(Variable(torch.LongTensor([x[1] for x in instance[3]]), volatile=True))

        target_variable = []
        all_relations = []
        p = 0
        for i in range(len(instance[3])):
            type, idx = instance[3][i]
            if idx == 5 or idx == 6:
                all_relations = all_relations + instance[4][p]
                all_relations.append([-2, 1])
                if len(instance[4][p]) == 0:
                    target_variable.append(None)
                elif use_cuda:
                    target_variable.append(Variable(torch.LongTensor([x[1] for x in instance[4][p]]), volatile=True).cuda(device))
                else:
                    target_variable.append(Variable(torch.LongTensor([x[1] for x in instance[4][p]]), volatile=True))
                p += 1
        dev_target_variables.append(target_variable)

        sel_gen_relations = []
        for type, idx in all_relations:
            if type == -2:
                sel_gen_relations.append(idx)
            else:
                sel_gen_relations.append(type+decoder.tags_info.tag_size)
        if use_cuda:
            dev_gold_variables.append(Variable(torch.LongTensor(sel_gen_relations), volatile=True).cuda(device))
        else:
            dev_gold_variables.append(Variable(torch.LongTensor(sel_gen_relations), volatile=True))
        assert p == len(instance[4])

        decoder.mask_pool.reset(len(instance[0]))
        mask = []
        p = 0
        for i in range(len(instance[3])):
            type, idx = instance[3][i]
            if idx == 5 or idx == 6:
                least = False
                if idx == 5 or (idx == 6 and instance[3][i+1][1] == 4):
                    least = True
                decoder.mask_pool.set_sdrs(idx == 5)
                temp_mask = decoder.mask_pool.get_all_mask(len(instance[4][p]), least)
                for k in range(len(instance[4][p])):
                    temp_idx = instance[4][p][k][0]
                    if temp_idx == -2:
                        temp_idx = instance[4][p][k][1]
                    else:
                        temp_idx += decoder.tags_info.tag_size
                    assert temp_mask[k][temp_idx] == decoder.mask_pool.need
                mask = mask + temp_mask
                p += 1
        assert p == len(instance[4])

        if use_cuda:
            dev_mask_variables.append(Variable(torch.FloatTensor(mask), volatile=True).cuda(device))
        else:
            dev_mask_variables.append(Variable(torch.FloatTensor(mask), volatile=True))


    for instance in dev_struct_instances:
        if use_cuda:
            dev_pred_struct_variables.append(Variable(torch.LongTensor([x[1] for x in instance]), volatile=True).cuda(device))
        else:
            dev_pred_struct_variables.append(Variable(torch.LongTensor([x[1] for x in instance]), volatile=True))

#======================================
    tst_sentence_variables = []
    tst_pred_struct_variables = []

    for instance in tst_instances:
        tst_sentence_variable = []
        if use_cuda:
            tst_sentence_variable.append(Variable(instance[0], volatile=True).cuda(device))
            tst_sentence_variable.append(Variable(instance[1], volatile=True).cuda(device))
            tst_sentence_variable.append(Variable(instance[2], volatile=True).cuda(device))
            
        else:
            tst_sentence_variable.append(Variable(instance[0], volatile=True))
            tst_sentence_variable.append(Variable(instance[1], volatile=True))
            tst_sentence_variable.append(Variable(instance[2], volatile=True))
            
        tst_sentence_variables.append(tst_sentence_variable)

    for instance in tst_struct_instances:
        if use_cuda:
            tst_pred_struct_variables.append(Variable(torch.LongTensor([x[1] for x in instance]), volatile=True).cuda(device))
        else:
            tst_pred_struct_variables.append(Variable(torch.LongTensor([x[1] for x in instance]), volatile=True))
#======================================
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
        loss = train(sentence_variables[idx][:3], sentence_variables[idx][3], target_variables[idx], gold_variables[idx], mask_variables[idx], encoder, s_encoder, decoder, encoder_optimizer, s_encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('epoch %.6f : %.10f' % (iter*1.0 / len(trn_instances), print_loss_avg))

        if iter % evaluate_every == 0:
            dev_idx = 0
            dev_loss = 0.0
            torch.save({"iter": iter, "idx":idx,  "encoder":encoder.state_dict(), "decoder":decoder.state_dict(), "s_encoder": s_encoder.state_dict(), "encoder_optimizer": encoder_optimizer.state_dict(), "decoder_optimizer": decoder_optimizer.state_dict(), "s_encoder_optimizer":s_encoder_optimizer.state_dict()}, model_dir+str(int(iter/evaluate_every))+".model")
            while dev_idx < len(dev_instances):
                if use_cuda:
                    torch.cuda.empty_cache()
                dev_loss += train(dev_sentence_variables[dev_idx][:3], dev_sentence_variables[dev_idx][3], dev_target_variables[dev_idx], dev_gold_variables[dev_idx], dev_mask_variables[dev_idx], encoder, s_encoder, decoder, encoder_optimizer, s_encoder_optimizer, decoder_optimizer, criterion, back_prop=False)
                dev_idx += 1
            print('dev loss %.10f' % (dev_loss/len(dev_instances)))
            evaluate(dev_sentence_variables, dev_pred_struct_variables, encoder, s_encoder, decoder, dev_out_dir+str(int(iter/evaluate_every))+".drs")
            evaluate(tst_sentence_variables, tst_pred_struct_variables, encoder, s_encoder, decoder, tst_out_dir+str(int(iter/evaluate_every))+".drs")

def evaluate(sentence_variables, pred_struct_variables, encoder, s_encoder, decoder, path):
    out = open(path,"w")
    for idx in range(len(sentence_variables)):
        if use_cuda:
            torch.cuda.empty_cache()
        
        tokens = decode(sentence_variables[idx][:3], pred_struct_variables[idx], encoder, s_encoder, decoder)

        structs = pred_struct_variables[idx].view(-1).data.tolist()

        p = 0
        output = []
        for i in range(len(structs)):
            output.append(decoder.tags_info.ix_to_tag[structs[i]])
            if structs[i] == 5 or structs[i] == 6:
                for type, tok in tokens[p]:
                    if type < 0 and tok == 1:
                        pass
                    else:
                        if type >= 0:
                            output.append(decoder.tags_info.ix_to_lemma[tok])
                        else:
                            output.append(decoder.tags_info.ix_to_tag[tok])
                        output.append(")")
                p += 1
        assert p == len(tokens)
        out.write(" ".join(output)+"\n")
        out.flush()
    out.close()
#####################################################################################
#####################################################################################
#####################################################################################
# main

from utils import readfile
from utils import data2instance_structure_relation
from utils import readpretrain
from utils import readstructure
from utils import structure2instance
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
dev_struct_file = "dev.struct"
tst_struct_file = "test.struct"
#dev_struct_file = "dev.struct.part"
#tst_struct_file = "test.struct.part"
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
mask_pool = RelationMask(tags_info)
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

dev_struct_data = readstructure(dev_struct_file)
tst_struct_data = readstructure(tst_struct_file)

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
structure_encoder = SimpleRNN(tags_info.tag_size, TAG_DIM, ENCODER_HIDDEN_DIM, n_layers=2, dropout_p=0.1)

###########################################################
# prepare training instance
trn_instances = data2instance_structure_relation(trn_data, [(word_to_ix,0), (pretrain_to_ix,0), (lemma_to_ix,0), tags_info])
print "trn size: " + str(len(trn_instances))
###########################################################
# prepare development instance
dev_instances = data2instance_structure_relation(dev_data, [(word_to_ix,0), (pretrain_to_ix,0), (lemma_to_ix,0), tags_info])
dev_struct_instances = structure2instance(dev_struct_data, tags_info)
print "dev size: " + str(len(dev_instances))
###########################################################
# prepare test instance
tst_instances = data2instance_structure_relation(tst_data, [(word_to_ix,0), (pretrain_to_ix,0), (lemma_to_ix,0), tags_info])
tst_struct_instances = structure2instance(tst_struct_data, tags_info)
print "tst size: " + str(len(tst_instances))

print "GPU", use_cuda
if use_cuda:
    encoder = encoder.cuda(device)
    attn_decoder = attn_decoder.cuda(device)
    structure_encoder = structure_encoder.cuda(device)

trainIters(trn_instances, dev_instances, tst_instances, dev_struct_instances, tst_struct_instances, encoder, structure_encoder, attn_decoder, print_every=1000, evaluate_every=50000, learning_rate=0.0005)

