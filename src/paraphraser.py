from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models import *
from helper import *
from utils  import *

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

import pdb

import abc
import six
import pdb

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

class Paraphraser(Model):

    def get_dataset(self, source, target):
        '''
        Input: source and target textLine datasets
        Output: Iterator for transformed, batched and zipped dataset
        '''
        source     = source.map(lambda sent: tf.string_split(["<sos> " + sent]).values)
        target_inp = target.map(lambda sent: tf.string_split(["<sos> " + sent ]).values)
        target_out = target.map(lambda sent: tf.string_split([sent  + " <eos>"]).values)

        #vocab lookup table
        dataset  = tf.data.Dataset.zip((source, target_inp, target_out))
        #Filter sents greater than 15
        dataset  = dataset.filter(lambda orig, para_inp, para_out: tf.logical_and(tf.logical_and(
                                                                    tf.greater(20, tf.size(orig)), tf.greater(20, tf.size(para_inp))), tf.greater(20, tf.size(para_out))))
        #lookup indexes
        dataset  = dataset.map(lambda orig, para_inp, para_out: ((self.vocab_table.lookup(orig), tf.size(orig)),
                                                                 (self.vocab_table.lookup(para_inp), tf.size(para_inp)),
                                                                 (self.vocab_table.lookup(para_out), tf.size(para_out))))

        dataset  = dataset.padded_batch(self.p.batch_size,  padded_shapes=((tf.TensorShape([None]), tf.TensorShape([])),
                                                                            (tf.TensorShape([None]), tf.TensorShape([])),
                                                                            (tf.TensorShape([None]), tf.TensorShape([]))),
                                                            padding_values=((tf.cast(0, tf.int64), 0), (tf.cast(0, tf.int64), 0), (tf.cast(0, tf.int64), 0)),
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
        self.vocab_table = tf.contrib.lookup.index_table_from_tensor(mapping=tf.constant(self.vocab), num_oov_buckets=0, default_value=0)
        self.id2w        = {idx: self.vocab[idx] for idx in range(len(self.vocab))}

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

    def build_single_cell(self):
        if (self.p.cell_type.lower() == 'gru'): cell_type = GRUCell
        else:                                   cell_type = LSTMCell

        cell = cell_type(self.p.hidden_size)

        if self.p.use_dropout:  cell = DropoutWrapper(cell, dtype=self.p.dtype, output_keep_prob=self.keep_prob_placeholder)    #change this
        if self.p.use_residual: cell = ResidualWrapper(cell)

        return cell


    def build_enc_cell(self, decode=False):
        return MultiRNNCell([self.build_single_cell() for i in range(self.p.depth)])

    def build_dec_cell(self):
        if self.use_beam_search:    #then tile
            self.logger.info("using beam search decoding")
            enc_outputs         = seq2seq.tile_batch(self.enc_outputs, multiplier=self.p.beam_width)
            enc_last_state      = nest.map_structure( lambda s: seq2seq.tile_batch(s, self.p.beam_width), self.enc_last_state)
            enc_inputs_length   = seq2seq.tile_batch(self.enc_inp_len, self.p.beam_width)

        else: #TRAINING
            enc_outputs     = self.enc_outputs
            enc_last_state      = self.enc_last_state
            enc_inputs_length   = self.enc_inp_len

        if self.p.attention_type.lower() == 'luong':
            self.attention_mechanism = attention_wrapper.LuongAttention(   num_units=self.p.hidden_size, memory=enc_outputs, memory_sequence_length=enc_inputs_length)
        else:
            self.attention_mechanism = attention_wrapper.BahdanauAttention(num_units=self.p.hidden_size, memory=enc_outputs, memory_sequence_length=enc_inputs_length)

        def attn_dec_input_fn(inputs, attention):
            if not self.p.attn_input_feeding:
                return inputs
            else:
                _input_layer = Dense(self.p.hidden_size, dtype=self.p.dtype, name='attn_input_feeding')
                return _input_layer(tf.concat([inputs, attention], -1))

        self.dec_cell_list     = [self.build_single_cell() for i in range(self.p.depth)]
        self.dec_cell_list[-1] = attention_wrapper.AttentionWrapper(cell                 = self.dec_cell_list[-1],
                                                                    attention_mechanism  = self.attention_mechanism,
                                                                    attention_layer_size = self.p.hidden_size,
                                                                    cell_input_fn        = attn_dec_input_fn,
                                                                    initial_cell_state   = enc_last_state[-1],
                                                                    alignment_history    = False,
                                                                    name                 = 'attention_wrapper')

        batch_size           = self.p.batch_size if not self.use_beam_search else self.p.batch_size*self.p.beam_width
        initial_state        = [state for state in enc_last_state]
        initial_state[-1]    = self.dec_cell_list[-1].zero_state(batch_size=batch_size, dtype=self.p.dtype)
        dec_initial_state    = tuple(initial_state)

        return MultiRNNCell(self.dec_cell_list), dec_initial_state


    def add_model(self):
        with tf.variable_scope("embed_lookup"):
            #modify initializer here to add glove/word2vec
            _wrd_embed    = tf.get_variable('embed_matrix',   [len(self.vocab)-1,  self.p.embed_dim],
                                            initializer=tf.contrib.layers.xavier_initializer(), regularizer=self.regularizer)

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

        self.logger.info("Building decoder")
        with tf.variable_scope("decoder"):
            #------------------------------------------------------------------THIS PART IS FOR TRAINING ONLY----------------------------------------------------------------------
            self.dec_cell, self.dec_initial_state = self.build_dec_cell()

            input_layer     = Dense(self.p.hidden_size, name="input_projection")
            output_layer    = Dense(len(self.vocab),    name="output_projection")

            if self.p.mode == 'train':
                self.dec_inp_embed = input_layer(self.dec_inp_embed) #decoder inputs dim should match encoder outputs dim
                training_helper    = seq2seq.TrainingHelper(inputs=self.dec_inp_embed, sequence_length=self.dec_inp_len, time_major=False, name='training_helper')
                training_decoder   = seq2seq.BasicDecoder(cell=self.dec_cell, helper=training_helper, initial_state=self.dec_initial_state, output_layer=output_layer)
                max_decoder_length = tf.reduce_max(self.dec_inp_len)

                (self.dec_outputs_train, self.dec_last_state_train, self.dec_outputs_length_train) = (seq2seq.dynamic_decode(decoder            = training_decoder,
                                                                                                                             output_time_major  = False,
                                                                                                                             impute_finished    = True,
                                                                                                                             maximum_iterations = max_decoder_length)
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

            #------------------------------------------------------------------THIS PART IS FOR DECODING ONLY----------------------------------------------------------------------
            self.logger.info("building decoder for inference")
            start_tokens = tf.ones([self.p.batch_size], tf.int32) * tf.cast(self.vocab_table.lookup(tf.constant('<sos>')), tf.int32)
            # pdb.set_trace()
            end_token    = tf.cast(self.vocab_table.lookup(tf.constant('<eos>')), tf.int32)

            def embed_and_input_proj(inputs):
                return input_layer(tf.nn.embedding_lookup(self.embed_matrix, inputs))

            if not self.use_beam_search:
                self.logger.info("Building greedy decoder")

                decoding_helper     = seq2seq.GreedyEmbeddingHelper(start_tokens = start_tokens,
                                                                        end_token    = end_token,
                                                                        embedding    = embed_and_input_proj)

                inference_decoder       = seq2seq.BasicDecoder( cell            = self.dec_cell,
                                                                helper          = decoding_helper,
                                                                initial_state   = self.dec_initial_state,
                                                                output_layer    = output_layer)

            else:
                self.logger.info("Building beam search decoder")

                self.inputs_placeholder = tf.placeholder(tf.float32, shape=[self.p.batch_size, self.p.beam_width])

                inference_decoder = beam_search_decoder.BeamSearchDecoder(  cell          = self.dec_cell,
                                                                            embedding     = embed_and_input_proj,
                                                                            start_tokens  = start_tokens,
                                                                            end_token     = end_token,
                                                                            initial_state = self.dec_initial_state,
                                                                            beam_width    = self.p.beam_width,
                                                                            output_layer  = output_layer)

            (self.dec_out_decode, self.dec_last_state_decode,
             self.dec_out_length_decode) = (seq2seq.dynamic_decode( inference_decoder_para, output_time_major=False, maximum_iterations=self.max_decode_step))

            if not self.use_beam_search_decode:
                #batchsize X seq_len X 1
                self.dec_pred_decode = tf.expand_dims(self.dec_out_decode.sample_id, -1)
            else:
                #batch_size X seq_len X beam_width
                self.dec_pred_decode = self.dec_out_decode.predicted_ids

    def get_accuracy(self):
        ''' Implement BLEU scores here
        '''
        pass

    def add_loss_op(self, loss):
        # if self.p.regularizer != None:
        #   loss += tf.contrib.layers.apply_regularization(self.p.regularizer, tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
          return loss

    def add_optimizer(self, loss):
        with tf.name_scope('Optimizer'):
            if self.p.opt == 'adam':  optimizer = tf.train.AdamOptimizer(self.p.lr)
            else:                     optimizer = tf.train.GradientDescentOptimizer(self.p.lr)

            train_op  = optimizer.minimize(loss)

        return train_op

    def __init__(self, params):
        self.p = params
        self.logger = get_logger(self.p.name, self.p.log_dir, self.p.config_dir)
        self.log_db = MongoClient('mongodb://10.24.28.104:27017/')[self.p.log_db][self.p.log_db]
        self.logger.info(vars(self.p)); pprint(vars(self.p))

        if self.p.l2 == 0.0: self.regularizer = None
        else:        self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.p.l2)

        self.load_data()

        self.use_beam_search = False
        if self.p.use_beam_search and self.p.mode == 'decode':
            self.use_beam_search = True

        self.add_model()

        if self.p.mode == 'train':
            self.loss       = self.add_loss_op(self.loss)
            self.train_op   = self.add_optimizer(self.loss)

        self.accuracy   = self.get_accuracy()


        self.merged_summ = tf.summary.merge_all()
        self.summ_writer = None

        self.min_loss = 1e10


    def pred2txt(self, preds, f):
        ''' preds of shape B X T X BW
        '''
        for i in range(preds.shape[0]):
            for beam in range(preds.shape[2]):
                txt = ' '.join([self.id2w[ids] for ids in preds[i, :, beam]])
                f.write(txt+'\n')
            f.write('\n')

    def evaluate(self, sess):
        sess.run(self.val_init)

        with open('output.txt', 'w') as f:
            while True:
                try:
                    dec_out = sess.run([self.dec_pred_decode])
                    self.pred2txt(dec_out[0], f)
                    #write code to evaluate BLEU score here

                except tf.errors.OutOfRangeError:
                    break

    def run_epoch(self, sess, epoch):
        # Training step
        sess.run(self.train_init)

        ep_loss = 0.0
        step = 0
        while True:
            try:
                _, loss = sess.run([self.train_op, self.loss])
                ep_loss += loss
                step    += 1

            except tf.errors.OutOfRangeError:
                break

            if step % self.p.patience == 0:
                self.logger.info("{} E: {} S: {} step_loss: {:.4f}".format(self.p.name, epoch, step, loss))

        if ep_loss < self.min_loss:
            self.saver.save(sess=sess, save_path=self.save_path)
            self.min_loss = ep_loss

        try:
            self.log_db.update_one(
                {'_id': self.p.name},
                {'$set': {
                            'train_loss':       float(ep_loss),
                            'Params':           vars(self.p),
                }}, upsert=True)

        except Exception as e:
            print('\nMongo ERROR Exception Cause: {}'.format(e.args[0]))

        self.logger.info('E:{} {} tr_loss: {:.3f} '.format(epoch, self.p.name, ep_loss))

    def fit(self, sess):
        self.summ_writer   = tf.summary.FileWriter("tf_board/Paraphraser/" + self.p.name, sess.graph)
        self.saver         = tf.train.Saver()
        save_dir           = 'checkpoints/' + self.p.name + '/'

        if not os.path.exists(save_dir): os.makedirs(save_dir)
        self.save_path     = os.path.join(save_dir, 'best_val_loss')

        self.best_val_loss  = 0.0

        if self.p.restore:
            self.saver.restore(sess, self.save_path)

        for epoch in range(self.p.max_epochs):
            self.run_epoch(sess, epoch)
            self.evaluate(sess)

if __name__== "__main__":

    parser = argparse.ArgumentParser(description='WORD GCN')

    # parser.add_argument('-data',     dest="data",     default='/scratchd/home/shikhar/gcn_word_embed/new_data/main_full_gcn.txt', help='Dataset to use')
    parser.add_argument('-data',        dest="data",       default='cora',         help='Dataset to use')
    parser.add_argument('-gpu',         dest="gpu",        default='0',            help='GPU to use')
    parser.add_argument('-name',        dest="name",       default='test',         help='Name of the run')
    parser.add_argument('-mode',        dest="mode",       default='train',        help='train/decode')

    parser.add_argument('-lr',          dest="lr",         default=0.01,   type=float,     help='Learning rate')
    parser.add_argument('-epoch',       dest="max_epochs",     default=1,    type=int,       help='Max epochs')
    parser.add_argument('-l2',          dest="l2",         default=0.01,   type=float,     help='L2 regularization')
    parser.add_argument('-seed',        dest="seed",       default=1234,   type=int,       help='Seed for randomization')
    parser.add_argument('-opt',         dest="opt",        default='adam',         help='Optimizer to use for training')
    parser.add_argument('-drop',        dest="dropout",    default=0,      type=float,     help='Dropout for full connected layer. Add support.')
    parser.add_argument('-batch_size',      dest="batch_size",     default=32,     type=int,       help='batch size to use')
    parser.add_argument('-dtype',         dest="dtype",    default=tf.float32,         help='Optimizer to use for training')

    parser.add_argument('-embed_dim',      dest="embed_dim",       default=64,     type=int,       help='embedding dimensions to use')

    parser.add_argument('-dump',        dest="dump",       action='store_true',        help='Dump results')
    parser.add_argument('-restore',     dest="restore",    action='store_true',        help='Restore from the previous best saved model')
    parser.add_argument('-log_db',      dest="log_db",     default='Paraphraser',      help='MongoDB database for dumping results')
    parser.add_argument('-logdir',      dest="log_dir",    default='/scratchd/home/shikhar/gcn_word_embed/src/log/',       help='Log directory')
    parser.add_argument('-config',      dest="config_dir",     default='../config/',       help='Config directory')
    parser.add_argument('-patience',    dest="patience",       default=10,     type=int,       help='how often to log output')

    #Model parameters
    parser.add_argument('-cell_type',       dest="cell_type",       default='lstm',         help='lstm or gru?')
    parser.add_argument('-hidden_size',     dest="hidden_size",     default=128,      type=int,     help='Hidden dimensions of the enc/dec')
    parser.add_argument('-depth',           dest="depth",           default=1,       type=int,      help='depth of enc/dec cells')
    parser.add_argument('-use_dropout',     dest="use_dropout",     action='store_true',        help='')
    parser.add_argument('-use_residual',    dest="use_residual",    action='store_true',        help='res connections?')
    parser.add_argument('-beam_width',      dest="beam_width",      default=10,      type=int,      help='')
    parser.add_argument('-attention_type',      dest="attention_type",      default='bahdanau',         help='luong/bahdanau')
    parser.add_argument('-attn_input_feeding',  dest="attn_input_feeding",  action='store_true',        help='')
    parser.add_argument('-use_beam_search',     dest="use_beam_search",     action='store_true',        help='')
    parser.add_argument('-max_decode_step',     dest="max_decode_step",     default=20,       type=int,      help='depth of enc/dec cells')


    args = parser.parse_args()

    if not args.restore: args.name = args.name + '_' + time.strftime("%d_%m_%Y") + '_' + time.strftime("%H:%M:%S")

    tf.set_random_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    set_gpu(args.gpu)

    model = Paraphraser(args)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.operation_timeout_in_ms=60000
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        tf.tables_initializer().run()
        model.fit(sess)

    print('Model Trained Successfully!!')