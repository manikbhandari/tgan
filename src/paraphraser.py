from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from helper import *
from cal_bleu import compute_bleu
from gan_train import create_model

import tensorflow as tf, time
from collections import Counter

import math

import numpy as np
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq

from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell
from tensorflow.python.ops.rnn_cell import DropoutWrapper, ResidualWrapper

from tensorflow.nn.rnn_cell import LSTMStateTuple as LSTMStateTuple

from tensorflow.python.ops import array_ops
from tensorflow.python.layers.core import Dense
from tensorflow.python.util import nest

from tensorflow.contrib.seq2seq.python.ops import attention_wrapper
from tensorflow.contrib.seq2seq.python.ops import beam_search_decoder

import os, glob, gzip, time
import logging

from datetime import datetime
from collections import OrderedDict
from collections import namedtuple

import ipdb as pdb

import abc
import six

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest

import tensorflow as tf



class Paraphraser():

    def debug(self, var_list):
        self.sess.run(tf.global_variables_initializer())
        tf.tables_initializer().run()
        self.sess.run(self.train_init)
        return self.sess.run(var_list)

    def get_dataset(self, source, target):
        '''
        Input: source and target textLine datasets
        Output: Iterator for transformed, batched and zipped dataset
        '''
        source     = source.map(lambda sent: tf.string_split(["<sos> " + sent + " <eos>"]).values)
        target_inp = target.map(lambda sent: tf.string_split(["<sos> " + sent ]).values)
        target_out = target.map(lambda sent: tf.string_split([sent  + " <eos>"]).values)

        #vocab lookup table
        dataset  = tf.data.Dataset.zip((source, target_inp, target_out))
        #Filter sents greater than 15
        dataset  = dataset.filter(lambda orig, para_inp, para_out: tf.logical_and(tf.logical_and(
                                                                    tf.greater(20, tf.size(orig)), tf.greater(20, tf.size(para_inp))),
                                                                    tf.greater(20, tf.size(para_out))))
        #lookup indexes
        dataset  = dataset.map(lambda orig, para_inp, para_out: ((self.vocab_table.lookup(orig), tf.size(orig)),
                                                                 (self.vocab_table.lookup(para_inp), tf.size(para_inp)),
                                                                 (self.vocab_table.lookup(para_out), tf.size(para_out))))

        dataset  = dataset.padded_batch(self.p.batch_size,  padded_shapes=((tf.TensorShape([None]), tf.TensorShape([])),
                                                                            (tf.TensorShape([None]), tf.TensorShape([])),
                                                                            (tf.TensorShape([None]), tf.TensorShape([]))),
                                                            padding_values=((tf.cast(self.w2id['<eos>'], tf.int64), self.w2id['<eos>']),
                                                                            (tf.cast(self.w2id['<eos>'], tf.int64), self.w2id['<eos>']),
                                                                            (tf.cast(self.w2id['<eos>'], tf.int64), self.w2id['<eos>'])),
                                                            drop_remainder=True
                                        )
        dataset.prefetch(2)

        return dataset

    def load_data(self):
        '''
        Adds iterators on the datasets
        '''
        self.id2w        = json.load(open('../data/quora/id2w.json'))
        self.w2id        = json.load(open('../data/quora/w2id.json'))

        self.vocab       = sorted([wrd for idx, wrd in self.id2w.items()])

        self.vocab[self.vocab.index("<eos>")] = '!'
        self.vocab[0]    = "<eos>"

        self.vocab_table = tf.contrib.lookup.index_table_from_tensor(mapping=tf.constant(self.vocab), num_oov_buckets=0, default_value=self.w2id['<unk>'])
        self.id2w        = {idx: self.vocab[idx] for idx in range(len(self.vocab))}
        self.w2id        = {wrd: idx for idx, wrd in self.id2w.items()}

        print("loading data")
        #Load tokenized data
        orig_sents = tf.data.TextLineDataset('../data/quora/train_split_100000_orig.txt')
        para_sents = tf.data.TextLineDataset('../data/quora/train_split_100000_para.txt')

        val_orig_sents = tf.data.TextLineDataset('../data/quora/val_fortrain_100000_orig.txt')
        val_para_sents = tf.data.TextLineDataset('../data/quora/val_fortrain_100000_para.txt')

        self.train_dataset = self.get_dataset(orig_sents, para_sents)
        self.val_dataset   = self.get_dataset(val_orig_sents, val_para_sents)

        self.iterator = tf.data.Iterator.from_structure(self.train_dataset.output_types, self.train_dataset.output_shapes)

        self.train_init = self.iterator.make_initializer(self.train_dataset)
        self.val_init   = self.iterator.make_initializer(self.val_dataset)


        (self.enc_inp, self.enc_inp_len), (self.dec_inp, self.dec_inp_len), (self.dec_out, self.dec_out_len)  = self.iterator.get_next()
        # sess.run(tf.global_variables_initializer())
        # tf.tables_initializer().run()
        # sess.run(self.val_init)
        # enc_inp = sess.run(self.enc_inp)
        # pdb.set_trace()

    def build_single_cell(self, hidden_size=None):
        if (self.p.cell_type.lower() == 'gru'): cell_type = GRUCell
        else:                                   cell_type = LSTMCell

        if not hidden_size: hidden_size = self.p.hidden_size
        cell = cell_type(hidden_size)

        if self.p.use_dropout:  cell = DropoutWrapper(cell, dtype=self.p.dtype, output_keep_prob=self.p.keep_prob)    #change this
        if self.p.use_residual: cell = ResidualWrapper(cell)

        return cell


    def build_enc_cell(self, decode=False):
        return MultiRNNCell([self.build_single_cell() for i in range(self.p.depth)])

    def build_bi_enc_cell(self):
        '''returns forward and backward cells for bidirectional stacked rnn'''
        fw_cells = [self.build_single_cell() for i in range(self.p.depth)]
        bw_cells = [self.build_single_cell() for i in range(self.p.depth)]
        return fw_cells, bw_cells

    def build_dec_cell(self, hidden_size):
        enc_outputs         = self.enc_outputs
        enc_last_state      = self.enc_last_state
        enc_inputs_length   = self.enc_inp_len

        if self.use_beam_search:
            self.logger.info("using beam search decoding")
            enc_outputs         = seq2seq.tile_batch(self.enc_outputs, multiplier=self.p.beam_width)
            enc_last_state      = nest.map_structure( lambda s: seq2seq.tile_batch(s, self.p.beam_width), self.enc_last_state)
            enc_inputs_length   = seq2seq.tile_batch(self.enc_inp_len, self.p.beam_width)

        if self.p.attention_type.lower() == 'luong':
            self.attention_mechanism = attention_wrapper.LuongAttention(   num_units=hidden_size, memory=enc_outputs, memory_sequence_length=enc_inputs_length)
        else:
            self.attention_mechanism = attention_wrapper.BahdanauAttention(num_units=hidden_size, memory=enc_outputs, memory_sequence_length=enc_inputs_length)

        def attn_dec_input_fn(inputs, attention):
            if not self.p.attn_input_feeding:
                return inputs
            else:
                _input_layer = Dense(hidden_size, dtype=self.p.dtype, name='attn_input_feeding')
                return _input_layer(tf.concat([inputs, attention], -1))

        self.dec_cell_list = [self.build_single_cell(hidden_size) for _ in range(self.p.depth)]

        if self.p.use_attn:
            self.dec_cell_list[-1] = attention_wrapper.AttentionWrapper(cell                 = self.dec_cell_list[-1],
                                                                        attention_mechanism  = self.attention_mechanism,
                                                                        attention_layer_size = hidden_size,
                                                                        cell_input_fn        = attn_dec_input_fn,
                                                                        initial_cell_state   = enc_last_state[-1],
                                                                        alignment_history    = False,
                                                                        name                 = 'attention_wrapper')

        batch_size           = self.p.batch_size if not self.use_beam_search else self.p.batch_size*self.p.beam_width
        initial_state        = [state for state in enc_last_state]
        if self.p.use_attn:
            initial_state[-1]    = self.dec_cell_list[-1].zero_state(batch_size=batch_size, dtype=self.p.dtype)
        dec_initial_state    = tuple(initial_state)

        return MultiRNNCell(self.dec_cell_list), dec_initial_state


    def sample_z(self):
        z_batch = np.random.normal(size=(self.p.batch_size, self.p.z_dim)).astype('float32')
        return z_batch

    def build_generator(self, hidden_size):
        generator_input = tf.random_normal(shape=[self.p.batch_size, self.p.z_dim])
        self.logger.info("Building generator...")

        with tf.variable_scope("generator"):
            layer_out = []
            layer_out.append(tf.contrib.layers.fully_connected(generator_input, self.p.generator_layer_units[0], trainable=False))

            for i, layer in enumerate(self.p.generator_layer_units[1:]):
                layer_out.append(tf.contrib.layers.fully_connected(layer_out[i], layer, trainable=False))

            perturbation = tf.contrib.layers.fully_connected(layer_out[len(layer_out)-1], hidden_size, activation_fn=None, trainable=False)
        return perturbation

    def add_model(self):
        with tf.variable_scope("embed_lookup"):
            #modify initializer here to add glove/word2vec
            embedding     = getGlove([wrd for wrd in self.vocab if wrd != '<unk>'], 'wiki_300')
            _wrd_embed    = tf.get_variable('embed_matrix',   [len(self.vocab)-1,  self.p.embed_dim], initializer=tf.constant_initializer(embedding), regularizer=self.regularizer)

            wrd_pad       = tf.Variable(tf.zeros([1, self.p.embed_dim]), trainable=False)
            self.embed_matrix = tf.concat([_wrd_embed, wrd_pad], axis=0)

        #Embed the source and target sentences. Elmo can be added here
        self.enc_inp_embed = tf.nn.embedding_lookup(self.embed_matrix, self.enc_inp)
        self.dec_inp_embed = tf.nn.embedding_lookup(self.embed_matrix, self.dec_inp)

        self.logger.info("Building encoder")
        with tf.variable_scope('encoder'):
            self.enc_cell                         = self.build_enc_cell()
            self.enc_outputs, self.enc_last_state = tf.nn.dynamic_rnn(  cell            = self.enc_cell,
                                                                        inputs          = self.enc_inp_embed,
                                                                        sequence_length = self.enc_inp_len,
                                                                        dtype           = self.p.dtype,
                                                                        time_major      = False,
                                                                        scope           = 'enc_rnn')

            #DED part. Also used for get_hidden
            self.enc_outputs_para, self.enc_last_state_para = tf.nn.dynamic_rnn(    cell            = self.enc_cell,
                                                                                    inputs          = self.dec_inp_embed,
                                                                                    sequence_length = self.dec_inp_len,
                                                                                    dtype           = self.p.dtype,
                                                                                    time_major      = False,
                                                                                    scope           = 'enc_rnn')
            if self.p.use_bidir:
                self.fw_cell, self.bw_cell = self.build_bi_enc_cell()
                self.enc_outputs, self.enc_last_state_fw, self.enc_last_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                                                                                        cells_fw        = self.fw_cell,
                                                                                        cells_bw        = self.bw_cell,
                                                                                        inputs          = self.enc_inp_embed,
                                                                                        sequence_length = self.enc_inp_len,
                                                                                        dtype           = self.p.dtype,
                                                                                        time_major      = False,
                                                                                        scope           = 'bi_enc_rnn')
                enc_last_state_fw = [state for state in self.enc_last_state_fw]
                enc_last_state_bw = [state for state in self.enc_last_state_bw]
                enc_last_state    = []
                for st, _ in enumerate(enc_last_state_fw):
                    enc_last_state.append(LSTMStateTuple(tf.concat([enc_last_state_fw[st].c, enc_last_state_bw[st].c], axis=-1), tf.concat([enc_last_state_fw[st].h, enc_last_state_bw[st].h], axis=-1)))
                self.enc_last_state = tuple(enc_last_state)

                self.enc_outputs_para, self.enc_last_state_fw_para, self.enc_last_state_bw_para = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                                                                                                            cells_fw        = self.fw_cell,
                                                                                                            cells_bw        = self.bw_cell,
                                                                                                            inputs          = self.dec_inp_embed,
                                                                                                            sequence_length = self.dec_inp_len,
                                                                                                            dtype           = self.p.dtype,
                                                                                                            time_major      = False,
                                                                                                            scope           = 'bi_enc_rnn')
                enc_last_state_fw_para = [state for state in self.enc_last_state_fw_para]
                enc_last_state_bw_para = [state for state in self.enc_last_state_bw_para]
                enc_last_state    = []
                for st, _ in enumerate(enc_last_state_fw_para):
                    enc_last_state.append(LSTMStateTuple(tf.concat([enc_last_state_fw_para[st].c, enc_last_state_bw_para[st].c], axis=-1),
                                                         tf.concat([enc_last_state_fw_para[st].h, enc_last_state_bw_para[st].h], axis=-1)))

                self.enc_last_state_para = tuple(enc_last_state)

            if self.p.use_gan:
                self.transformation   = self.build_generator(self.p.hidden_size*2 if self.p.use_bidir else self.p.hidden_size)
                enc_last_state        = [state for state in self.enc_last_state]
                enc_last_state[-1]    = LSTMStateTuple(self.enc_last_state[-1].c, self.enc_last_state[-1].h + 10*self.transformation)
                self.enc_last_state   = tuple(enc_last_state)

        self.dec_cell, self.dec_initial_state = self.build_dec_cell(self.p.hidden_size*2 if self.p.use_bidir else self.p.hidden_size)

        self.input_layer                      = Dense(self.p.hidden_size*2 if self.p.use_bidir else self.p.hidden_size, name="input_projection")
        self.output_layer                     = Dense(len(self.vocab),    name="output_projection")

        if self.p.mode == 'train':
            self.logger.info("Building training decoder")

            self.dec_inp_embed = self.input_layer(self.dec_inp_embed) #decoder inputs dim should match encoder outputs dim

            training_helper         = seq2seq.TrainingHelper(inputs=self.dec_inp_embed, sequence_length=self.dec_inp_len, time_major=False, name='training_helper')
            training_decoder        = seq2seq.BasicDecoder(cell=self.dec_cell, helper=training_helper, initial_state=self.dec_initial_state, output_layer=self.output_layer)
            self.max_decoder_length = tf.reduce_max(self.dec_inp_len)
            # res = self.debug([self.dec_inp_embed]); pdb.set_trace()

            (self.dec_outputs_train, self.dec_last_state_train, self.dec_outputs_length_train) = (seq2seq.dynamic_decode(
                                                                                                     decoder            = training_decoder,
                                                                                                     output_time_major  = False,
                                                                                                     impute_finished    = True,
                                                                                                     maximum_iterations = self.max_decoder_length)
                                                                                                  )

            #since output layer is passed to decoder, logits = output
            self.dec_logits_train   = self.dec_outputs_train.rnn_output
            self.dec_pred_train     = tf.argmax(self.dec_logits_train, axis=-1, name='decoder_pred_train')
            masks                   = tf.sequence_mask(lengths=self.dec_inp_len, maxlen=tf.shape(self.dec_inp)[1], dtype=self.p.dtype, name='masks')

            self.loss = seq2seq.sequence_loss(logits                        = self.dec_logits_train,
                                                   targets                  = self.dec_out,
                                                   weights                  = masks,
                                                   average_across_timesteps = True,
                                                   average_across_batch     = True)

            tf.summary.scalar('loss', self.loss)

        elif self.p.mode == 'decode':

            self.logger.info("building decoder for inference")
            start_tokens = tf.ones([self.p.batch_size], tf.int32) * tf.cast(self.vocab_table.lookup(tf.constant('<sos>')), tf.int32)
            self.start_tokens = start_tokens
            # pdb.set_trace()
            end_token    = tf.cast(self.vocab_table.lookup(tf.constant('<eos>')), tf.int32)

            def embed_and_input_proj(inputs):
                return self.input_layer(tf.nn.embedding_lookup(self.embed_matrix, inputs))

            if not self.p.use_beam_search:
                self.logger.info("Building greedy decoder")

                decoding_helper     = seq2seq.GreedyEmbeddingHelper(start_tokens = start_tokens,
                                                                        end_token    = end_token,
                                                                        embedding    = embed_and_input_proj)

                inference_decoder       = seq2seq.BasicDecoder( cell            = self.dec_cell,
                                                                helper          = decoding_helper,
                                                                initial_state   = self.dec_initial_state,
                                                                output_layer    = self.output_layer)

            else:
                self.logger.info("Building beam search decoder")

                inference_decoder = beam_search_decoder.BeamSearchDecoder(  cell          = self.dec_cell,
                                                                            embedding     = embed_and_input_proj,
                                                                            start_tokens  = start_tokens,
                                                                            end_token     = end_token,
                                                                            initial_state = self.dec_initial_state,
                                                                            beam_width    = self.p.beam_width,
                                                                            output_layer  = self.output_layer)

            (self.dec_out_decode, self.dec_last_state_decode,
             self.dec_out_length_decode) = (seq2seq.dynamic_decode( inference_decoder, output_time_major=False, maximum_iterations=self.p.max_decode_step))

            if not self.p.use_beam_search:
                #batchsize X seq_len X 1
                self.dec_pred_decode = tf.expand_dims(self.dec_out_decode.sample_id, -1)
            else:
                #batch_size X seq_len X beam_width
                self.dec_pred_decode = self.dec_out_decode.predicted_ids


    def add_loss_op(self, loss):
        # if self.p.regularizer != None:
        #   loss += tf.contrib.layers.apply_regularization(self.p.regularizer, tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
          # loss += self.p.ded_lambda * tf.nn.l2_loss(self.enc_last_state[-1].h - self.enc_last_state_para[-1].h)
          return loss

    def add_optimizer(self, loss):
        with tf.name_scope('Optimizer'):
            if self.p.opt == 'adam':  optimizer = tf.train.AdamOptimizer(self.p.lr)
            else:                     optimizer = tf.train.GradientDescentOptimizer(self.p.lr)

            train_op  = optimizer.minimize(loss)

        return train_op

    def __init__(self, params, sess):
        self.sess = sess
        self.p    = params
        self.logger = get_logger(self.p.name, self.p.log_dir, self.p.config_dir)
        self.log_db = MongoClient('mongodb://10.24.28.104:27017/')[self.p.log_db][self.p.log_db]
        # if self.p.mode == 'train':
        #     self.logger.info(vars(self.p)); pprint(vars(self.p))

        if self.p.l2 == 0.0: self.regularizer = None
        else:        self.regularizer         = tf.contrib.layers.l2_regularizer(scale=self.p.l2)

        if self.p.use_beam_search and self.p.mode == 'decode':
            self.use_beam_search = True
        else:
            self.use_beam_search = False

        self.load_data()
        self.add_model()

        if self.p.mode == 'train':
            self.loss       = self.add_loss_op(self.loss)
            self.train_op   = self.add_optimizer(self.loss)

        self.merged_summ = tf.summary.merge_all()

        generator_vars     = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder/generator')
        all_vars           = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver         = tf.train.Saver(var_list=list(set(all_vars)-set(generator_vars)))
        self.save_dir           = 'checkpoints/' + self.p.name + '/'

        if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)
        self.save_path     = os.path.join(self.save_dir, 'best_val_loss')

        self.min_val_loss  = 1e10

    def pred2txt(self, preds, orig, references, f):
        ''' preds of shape B X T X BW
        '''
        remove_ids = [0, self.w2id['<sos>'], self.w2id['<eos>']]
        for i in range(preds.shape[0]):

            txt = ' '.join([self.id2w[ids] for ids in orig[i, :] if ids not in remove_ids])
            f.write(txt+'\n')

            txt = ' '.join([self.id2w[ids] for ids in references[i, :] if ids not in remove_ids])
            f.write(txt+'\n')

            f.write('-----------------------------------------\n')
            for beam in range(preds.shape[2]):
                txt = ' '.join([self.id2w[ids] for ids in preds[i, :, beam] if ids not in remove_ids])
                if '?' in txt: txt = txt[:txt.find('?')+1]
                f.write(txt+'\n')

            f.write('\n')


    def get_bleu(self, refs, hyps):
        ''' ref shape: B X T, hyp shape; B X T X BW
        '''
        remove_ids = [self.w2id['<sos>'], self.w2id['<eos>']]
        filt_hyps = []
        filt_refs = []

        for hyp in hyps:
            filt = [self.id2w[wrd] for wrd in hyp if wrd not in remove_ids]
            filt_hyps.append(filt)

        for ref in refs:
            filt_ = []
            for k in ref:
                filt = [self.id2w[wrd] for wrd in k if wrd not in remove_ids]
                filt_.append(filt)

            filt_refs.append(filt_)

        bleu = compute_bleu(filt_refs, filt_hyps)
        return bleu


    def sent2id(self, sent):
        remove_ids = [self.w2id['<sos>'], self.w2id['<eos>']]
        return [self.id2w[idx] for idx in sent if idx not in remove_ids]

    def get_transformations(self, sess):
        all_src, all_tgt, all_preds, all_transforms, all_src_vecs = [], [], [], [], []

        for num in range(self.p.num_transforms):
            sess.run(self.val_init)
            self.logger.info("gathering transformations for epoch {}".format(num))
            tmp_src, tmp_tgt, tmp_preds, tmp_transforms, tmp_src_vecs = [], [], [], [], []
            while True:
                try:
                    preds, enc_inp, dec_inp, src_vecs, transforms = sess.run([self.dec_pred_decode, self.enc_inp, self.dec_inp, self.enc_last_state[-1].h, self.transformation])
                    preds   = preds.reshape([self.p.batch_size, -1])

                    enc_inp = [self.sent2id(sent) for sent in enc_inp]
                    dec_inp = [self.sent2id(sent) for sent in dec_inp]
                    preds   = [self.sent2id(sent) for sent in preds]

                    tmp_src        += enc_inp
                    tmp_tgt        += dec_inp
                    tmp_preds      += preds
                    tmp_transforms += transforms.tolist()
                    tmp_src_vecs    += src_vecs.tolist()
                    break

                except tf.errors.OutOfRangeError:
                    break

            all_src.append(tmp_src)
            all_tgt.append(tmp_tgt)
            all_preds.append(tmp_preds)
            all_transforms.append(tmp_transforms)
            all_src_vecs.append(tmp_src_vecs)

        pdb.set_trace()
        all_preds_ = [list(el) for el in zip()]


    def evaluate(self, sess):
        sess.run(self.val_init)

        all_hyps, all_refs = [], []
        write_file = os.path.join('generations', self.p.out_file)
        if self.p.mode == 'decode':
            f = open(write_file, 'w', encoding='utf8')
            self.logger.info("writing predictions to {}".format(write_file))

        val_loss = []
        while True:
            try:
                if self.p.mode == 'train':
                    loss = sess.run([self.loss])[0]
                    val_loss.append(loss)
                else:
                    preds, enc_inp, dec_inp, dec_out, st_tok = sess.run([self.dec_pred_decode, self.enc_inp, self.dec_inp, self.dec_out, self.start_tokens])
                    flat_preds = preds.swapaxes(2, 1)
                    flat_preds = flat_preds.reshape(-1, flat_preds.shape[2]).tolist()
                    flat_refs  = np.repeat(dec_inp, self.p.beam_width, axis=0).tolist()
                    flat_refs  = [[ref] for ref in flat_refs]

                    all_refs += flat_refs
                    all_hyps += flat_preds

                    self.pred2txt(preds, enc_inp, dec_inp, f)

            except tf.errors.OutOfRangeError:
                break

        if self.p.mode == 'train':
            return np.average(val_loss)

        val_bleu = self.get_bleu(all_refs, all_hyps)
        return val_bleu[0]

    def get_hidden(self, sess):
        sess.run(self.train_init)
        inp_hid = []
        tar_hid = []

        self.logger.info("fetching hidden representations")
        while True:
            try:
                inp_hidden, tar_hidden, inp_len, tar_len = sess.run([self.enc_last_state, self.enc_last_state_para, self.enc_inp_len, self.dec_inp_len])

                inp_hidden = inp_hidden[-1].h
                tar_hidden = tar_hidden[-1].h

                for rep in inp_hidden:
                    inp_hid.append(list(rep))

                for rep in tar_hidden:
                    tar_hid.append(list(rep))

            except tf.errors.OutOfRangeError:
                break

        gan_train_data = {'train': []}
        for i in range(len(inp_hid)):
            gan_train_data['train'].append({
                            'inp_sent':'temp temp',
                            'inp_hidden': list(map(float, inp_hid[i])),
                            'tar_sent':'temp temp',
                            'tar_hidden': list(map(float, tar_hid[i]))
            })

        save_file = self.save_dir + self.p.gan_data_file
        self.logger.info("dumping {} representations to {}.".format(len(gan_train_data['train']), save_file))

        with open(save_file, 'w') as f:
            json.dump(gan_train_data, f)

    def run_epoch(self, sess, epoch):
        # Training step
        sess.run(self.train_init)

        tr_loss = 0
        step    = 0
        while True:
            try:
                _, loss  = sess.run([self.train_op, self.loss])
                tr_loss += loss
                step    += 1

            except tf.errors.OutOfRangeError:
                break

            if step % self.p.patience == 0:
                self.logger.info("{} E: {} S: {} step_loss: {:.4f}".format(self.p.name, epoch, step, loss))

        # run on validation data
        val_loss = self.evaluate(sess)

        if val_loss <= self.min_val_loss:
            self.saver.save(sess=sess, save_path=self.save_path)
            self.min_val_loss = val_loss

            try:
                self.log_db.update_one(
                    {'_id': self.p.name},
                        {
                            '$set': {
                                        'Params':           {k:v for k, v in vars(self.p).items() if k != 'dtype'},
                                    },
                         '$push':{
                                    'val_loss':         float(val_loss),
                                    'tr_loss':         float(tr_loss),
                                 }
                         }, upsert=True)

            except Exception as e:
                print('\nMongo ERROR Exception Cause: {}'.format(e.args[0]))

        self.logger.info('E:{} {} tr_loss:  {:.3f} '.format(epoch, self.p.name, tr_loss))
        self.logger.info('E:{} {} val_loss: {:.3f} '.format(epoch, self.p.name, val_loss))

    def fit(self, sess):

        if self.p.restore:
            #restore gan part of the model
            if self.p.use_gan:
                generator_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder/generator')
                gan_ckpt       = tf.train.latest_checkpoint(self.p.gan_path)
                if not gan_ckpt:
                    raise FileNotFoundError("gan checkpoint not found")

                gan_saver      = tf.train.Saver(var_list=generator_vars)
                gan_saver.restore(sess, gan_ckpt)

            self.saver.restore(sess, self.save_path)

        if self.p.mode == 'train':
            for epoch in range(self.p.max_epochs):
                self.run_epoch(sess, epoch)

        elif self.p.mode == 'decode':
            if self.p.use_div:
                self.get_transformations(sess)
            else:
                bleu = self.evaluate(sess)
                self.logger.info("BLEU: {}".format(bleu))

        elif self.p.mode == 'get_hidden':
            self.get_hidden(sess)

        else:
            raise NotImplementedError('mode must be train/decode/get_hidden. Found {}'.format(self.p.mode))

if __name__== "__main__":

    parser = argparse.ArgumentParser(description='Paraphraser')

    parser.add_argument('-gpu',         dest="gpu",        default='0',            help='GPU to use')
    parser.add_argument('-name',        dest="name",       default='test',         help='Name of the run')
    parser.add_argument('-mode',        dest="mode",       default='train',        help='train/decode')

    parser.add_argument('-lr',          dest="lr",         default=0.001,        type=float,     help='Learning rate')
    parser.add_argument('-epoch',       dest="max_epochs", default=100,       type=int,       help='Max epochs')
    parser.add_argument('-l2',          dest="l2",         default=0.01,        type=float,     help='L2 regularization')
    parser.add_argument('-seed',        dest="seed",       default=1234,        type=int,       help='Seed for randomization')
    parser.add_argument('-opt',         dest="opt",        default='adam',                      help='Optimizer to use for training')
    parser.add_argument('-drop',        dest="dropout",    default=0,           type=float,     help='Dropout for full connected layer. Add support.')
    parser.add_argument('-batch_size',  dest="batch_size", default=32,          type=int,       help='batch size to use')
    parser.add_argument('-dtype',       dest="dtype",      default=tf.float32,                  help='Optimizer to use for training')


    parser.add_argument('-dump',        dest="dump",       action='store_true',        help='Dump results')
    parser.add_argument('-restore',     dest="restore",    action='store_true',        help='Restore from the previous best saved model')
    parser.add_argument('-log_db',      dest="log_db",     default='Paraphraser',      help='MongoDB database for dumping results')
    parser.add_argument('-logdir',      dest="log_dir",    default='/scratchd/home/shikhar/gcn_word_embed/src/log/',       help='Log directory')
    parser.add_argument('-config',      dest="config_dir", default='../config/',       help='Config directory')
    parser.add_argument('-out_file',    dest="out_file",   default='output.txt',       help='Config directory')
    parser.add_argument('-patience',    dest="patience",   default=10, type=int,       help='how often to log output')

    #Model parameters
    parser.add_argument('-cell_type',           dest="cell_type",           default='lstm',                     help='lstm or gru?')
    parser.add_argument('-hidden_size',         dest="hidden_size",         default=256,        type=int,       help='Hidden dimensions of the enc/dec')
    parser.add_argument('-embed_dim',           dest="embed_dim",           default=300,        type=int,       help='embedding dimensions to use')
    parser.add_argument('-depth',               dest="depth",               default=2,          type=int,       help='depth of enc/dec cells')
    parser.add_argument('-use_dropout',         dest="use_dropout",         action='store_true',                help='')
    parser.add_argument('-keep_prob',           dest="keep_prob",           default=0.7,        type=float,     help='')
    parser.add_argument('-use_residual',        dest="use_residual",        action='store_true',                help='res connections?')
    parser.add_argument('-beam_width',          dest="beam_width",          default=10,         type=int,       help='')
    parser.add_argument('-attention_type',      dest="attention_type",      default='bahdanau',                 help='luong/bahdanau')
    parser.add_argument('-attn_input_feeding',  dest="attn_input_feeding",  action='store_true',                help='')
    parser.add_argument('-use_beam_search',     dest="use_beam_search",     action='store_true',                help='')
    parser.add_argument('-use_attn',            dest="use_attn",            action='store_true',                help='')
    parser.add_argument('-use_gan',             dest="use_gan",             action='store_true',                help='')
    parser.add_argument('-use_bidir',           dest="use_bidir",           action='store_true',                help='use bidirectional encoder rnn')
    parser.add_argument('-use_div',             dest="use_div",             action='store_true',                help='use bidirectional encoder rnn')
    parser.add_argument('-ded_lambda',          dest="ded_lambda",          default=1,                help='use bidirectional encoder rnn')
    parser.add_argument('-max_decode_step',     dest="max_decode_step",     default=20,          type=int,      help='depth of enc/dec cells')

    args = parser.parse_args()

    if not args.restore: args.name = args.name + '_' + time.strftime("%d_%m_%Y") + '_' + time.strftime("%H:%M:%S")

    # tf.set_random_seed(args.seed)
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    set_gpu(args.gpu)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.operation_timeout_in_ms  = 60000



    with tf.Session(config=config) as sess:
        model = Paraphraser(args, sess)
        with open(os.path.join(model.save_dir, 'params'), 'w') as f:
            params = vars(args)
            params = {k: v for k,v in params.items() if k != 'dtype'}
            json.dump(params, f)

        sess.run(tf.global_variables_initializer())
        tf.tables_initializer().run()
        model.fit(sess)


    print('Model Trained Successfully!!')