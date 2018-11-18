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
# from tensorflow.contrib.seq2seq.python.ops import beam_search_decoder
import beam_search_decoder
import dynamic_decode
import custom_dd

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

__all__ = ["Decoder", "dynamic_decode"]


_transpose_batch_time = rnn._transpose_batch_time  # pylint: disable=protected-access
_zero_state_tensors = rnn_cell_impl._zero_state_tensors  # pylint: disable=protected-access




def _create_zero_outputs(size, dtype, batch_size):
    """Create a zero outputs Tensor structure."""
    def _create(s, d):
        return _zero_state_tensors(s, batch_size, d)

    return nest.map_structure(_create, size, dtype)



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
                                                                    tf.greater(25, tf.size(orig)), tf.greater(25, tf.size(para_inp))), tf.greater(25, tf.size(para_out))))
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
        self.vocab           = sorted([wrd for idx, wrd in self.id2w.items()])
        self.vocab_table = tf.contrib.lookup.index_table_from_tensor(mapping=tf.constant(self.vocab), num_oov_buckets=0, default_value=0)

        self.id2w        = {idx: self.vocab[idx] for idx in range(len(self.vocab))}

        print("loading data")
        #Load tokenized data
        orig_sents = tf.data.TextLineDataset('../data/quora/train_split_100000_orig.txt')
        para_sents = tf.data.TextLineDataset('../data/quora/train_split_100000_para.txt')

        val_orig_sents = tf.data.TextLineDataset('../data/quora/val_fortrain_100000_orig.txt')
        val_para_sents = tf.data.TextLineDataset('../data/quora/val_fortrain_100000_para.txt')
        # pdb.set_trace()
        self.train_dataset = self.get_dataset(orig_sents, para_sents)
        self.val_dataset   = self.get_dataset(val_orig_sents, val_para_sents)

        self.iterator = tf.data.Iterator.from_structure(self.train_dataset.output_types, self.train_dataset.output_shapes)

        self.train_init = self.iterator.make_initializer(self.train_dataset)
        self.val_init   = self.iterator.make_initializer(self.val_dataset)


        (self.enc_inp, self.enc_inp_len), (self.dec_inp, self.dec_inp_len), (self.dec_out, self.dec_out_len)  = self.iterator.get_next()
        # (self.val_source, self.val_source_len), (self.val_target_inp, self.val_target_len), (self.val_target_out, self.val_target_len)  = self.val_iterator.get_next()

    def build_single_cell(self):
        if (self.p.cell_type.lower() == 'gru'): cell_type = GRUCell
        else:                   cell_type = LSTMCell
        cell = cell_type(self.p.hidden_size)

        if self.p.use_dropout:  cell = DropoutWrapper(cell, dtype=self.p.dtype, output_keep_prob=self.keep_prob_placeholder)    #change this
        if self.p.use_residual: cell = ResidualWrapper(cell)

        return cell


    def build_enc_cell(self, decode=False):
        return MultiRNNCell([self.build_single_cell() for i in range(self.p.depth)])

    def build_dec_cell(self):
        if self.use_beam_search:
            self.logger.info("using beam search decoding")
            enc_outputs     = seq2seq.tile_batch(self.enc_outputs, multiplier=self.p.beam_width)
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

        # dec_cell_list = [self.build_single_cell() for i in range(self.p.depth)]

        def attn_dec_input_fn(inputs, attention):
            if not self.p.attn_input_feeding:
                return inputs
            else:
                _input_layer = Dense(self.p.hidden_size, dtype=self.p.dtype, name='attn_input_feeding')
                return _input_layer(tf.concat([inputs, attention], -1))

        self.dec_cell_list = [self.build_single_cell() for i in range(self.p.depth)]
        self.dec_cell_list[-1] = attention_wrapper.AttentionWrapper(cell         = self.dec_cell_list[-1],
                                                                    attention_mechanism  = self.attention_mechanism,
                                                                    attention_layer_size = self.p.hidden_size,
                                                                    cell_input_fn    = attn_dec_input_fn,
                                                                    initial_cell_state   = enc_last_state[-1],
                                                                    alignment_history    = False,
                                                                    name         = 'attention_wrapper')

        batch_size       = self.p.batch_size if not self.use_beam_search else self.p.batch_size*self.p.beam_width
        initial_state    = [state for state in enc_last_state]
        initial_state[-1]    = self.dec_cell_list[-1].zero_state(batch_size=batch_size, dtype=self.p.dtype)
        dec_initial_state    = tuple(initial_state)

        return MultiRNNCell(self.dec_cell_list), dec_initial_state


    def add_model(self):
        with tf.variable_scope("embed_lookup"):
            _wrd_embed    = tf.get_variable('embed_matrix',   [len(self.vocab)-1,  self.p.embed_dim], initializer=tf.contrib.layers.xavier_initializer(), regularizer=self.regularizer)
            wrd_pad       = tf.Variable(tf.zeros([1, self.p.embed_dim]), trainable=False)
            self.embed_matrix = tf.concat([_wrd_embed, wrd_pad], axis=0)

        #Embed the source and target sentences
        self.enc_inp_embed = tf.nn.embedding_lookup(self.embed_matrix, self.enc_inp)
        self.dec_inp_embed = tf.nn.embedding_lookup(self.embed_matrix, self.dec_inp)

        self.logger.info("Building encoder")
        with tf.variable_scope('encoder'):
            self.enc_cell             = self.build_enc_cell()
            self.enc_outputs, self.enc_last_state = tf.nn.dynamic_rnn(  cell        = self.enc_cell,
                                                                        inputs      = self.enc_inp_embed,
                                                                        sequence_length = self.enc_inp_len,
                                                                        dtype       = self.p.dtype,
                                                                        time_major      = False,
                                                                        scope       = 'enc_rnn')

        self.logger.info("Building decoder")
        with tf.variable_scope("decoder"):
            #------------------------------------------------------------------THIS PART IS FOR TRAINING ONLY----------------------------------------------------------------------
            self.dec_cell, self.dec_initial_state = self.build_dec_cell()

            input_layer     = Dense(self.p.hidden_size, name="input_projection")
            output_layer    = Dense(len(self.vocab), name="output_projection")


            if self.p.mode == 'train':
                self.dec_inp_embed = input_layer(self.dec_inp_embed)
                training_helper   = seq2seq.TrainingHelper(inputs=self.dec_inp_embed, sequence_length=self.dec_inp_len, time_major=False, name='training_helper')
                training_decoder  = seq2seq.BasicDecoder(cell=self.dec_cell, helper=training_helper, initial_state=self.dec_initial_state, output_layer=output_layer)
                #max decode timesteps in current batch
                max_decoder_length = tf.reduce_max(self.dec_inp_len)

                (self.dec_outputs_train, self.dec_last_state_train, self.dec_outputs_length_train) = (seq2seq.dynamic_decode(decoder        = training_decoder,
                                                                                                                             output_time_major  = False,
                                                                                                                             impute_finished    = True,
                                                                                                                             maximum_iterations = max_decoder_length)
                                                                                                                            )

                #since output layer is passed to decoder, logits = output
                self.dec_logits_train   = tf.identity(self.dec_outputs_train.rnn_output)
                self.dec_pred_train     = tf.argmax(self.dec_logits_train, axis=-1, name='decoder_pred_train')
                # to_pad          = tf.maximum(0, tf.shape(self.dec_inp)[1] - max_decoder_length)
                masks           = tf.sequence_mask(lengths=self.dec_inp_len, maxlen=tf.shape(self.dec_inp)[1], dtype=self.p.dtype, name='masks')
                # paddings = tf.zeros(shape=[self.p.batch_size, to_pad, len(self.vocab)])
                # self.dec_logits_train = tf.concat([self.dec_logits_train, paddings], axis=1)
                self.loss = seq2seq.sequence_loss(logits            = self.dec_logits_train,
                                                       targets          = self.dec_out,
                                                       weights          = masks,
                                                       average_across_timesteps = True,
                                                       average_across_batch     = True)

                tf.summary.scalar('loss', self.loss)

            #------------------------------------------------------------------THIS PART IS FOR DECODING ONLY----------------------------------------------------------------------
            elif self.p.mode == 'decode':
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

                    self.dec_pred_decode = tf.expand_dims(self.dec_outputs_decode.sample_id, -1)        #batchsize X seq_len X 1

                else:
                    self.logger.info("Building beam search decoder")

                    self.inputs_placeholder = tf.placeholder(tf.float32, shape=[self.p.batch_size, self.p.beam_width])

                    self.inference_decoder = beam_search_decoder.BeamSearchDecoder(  cell       = self.dec_cell,
                                                                                embedding           = embed_and_input_proj,
                                                                                start_tokens    = start_tokens,
                                                                                end_token           = end_token,
                                                                                initial_state   = self.dec_initial_state,
                                                                                beam_width          = self.p.beam_width,
                                                                                output_layer    = output_layer)

                # (self.dec_outputs_decode, self.dec_last_state_decode, self.decodec_outputs_length_decdoe, self.next_outputs, self.ta) = (custom_dd.dynamic_decode(inference_decoder,
                #                                                                                                                       output_time_major = False,
                #                                                                                                                       maximum_iterations = self.p.max_decode_step))

                # if not self.use_beam_search:
                # else:
                #       self.dec_pred_decode = self.dec_outputs_decode.predicted_ids                        #batch_size X seq_len X beam_width

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
            else:             optimizer = tf.train.GradientDescentOptimizer(self.p.lr)

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

    def dynamic_decode(self, decoder, output_time_major=False, impute_finished=False, maximum_iterations=None, scope=None):

        with variable_scope.variable_scope(scope, "decoder") as varscope:
            # Determine context types.
            ctxt            = ops.get_default_graph()._get_control_flow_context()  # pylint: disable=protected-access
            is_xla          = control_flow_util.GetContainingXLAContext(ctxt) is not None
            in_while_loop   = (control_flow_util.GetContainingWhileContext(ctxt) is not None)

            if not context.executing_eagerly() and not in_while_loop:
                if varscope.caching_device is None:
                    varscope.set_caching_device(lambda op: op.device)

            if maximum_iterations is not None:
                maximum_iterations = ops.convert_to_tensor(maximum_iterations, dtype=dtypes.int32, name="maximum_iterations")
                if maximum_iterations.get_shape().ndims != 0:
                    raise ValueError("maximum_iterations must be a scalar")

            initial_finished, initial_inputs, initial_state = decoder.initialize()

            zero_outputs = _create_zero_outputs(decoder.output_size, decoder.output_dtype, decoder.batch_size)

            if is_xla and maximum_iterations is None:
                raise ValueError("maximum_iterations is required for XLA compilation.")

            if maximum_iterations is not None:
                initial_finished     = math_ops.logical_or(initial_finished, 0 >= maximum_iterations)

            initial_sequence_lengths = array_ops.zeros_like(initial_finished, dtype=dtypes.int32)
            initial_time             = constant_op.constant(0, dtype=dtypes.int32)

            def _shape(batch_size, from_shape):
                if (not isinstance(from_shape, tensor_shape.TensorShape) or from_shape.ndims == 0):
                    return tensor_shape.TensorShape(None)
                else:
                    batch_size = tensor_util.constant_value(ops.convert_to_tensor(batch_size, name="batch_size"))
                    return tensor_shape.TensorShape([batch_size]).concatenate(from_shape)

            dynamic_size = maximum_iterations is None or not is_xla

            def _create_ta(s, d):
                return tensor_array_ops.TensorArray(dtype=d, size=0 if dynamic_size else maximum_iterations, dynamic_size=dynamic_size, element_shape=_shape(decoder.batch_size, s))

            initial_outputs_ta = nest.map_structure(_create_ta, decoder.output_size, decoder.output_dtype)

            def body(time, outputs_ta, state, inputs, finished, sequence_lengths):

                (next_outputs, decoder_state, next_inputs, decoder_finished) = decoder.step(time, inputs, state)
                if decoder.tracks_own_finished:
                    next_finished     = decoder_finished
                else:
                    next_finished     = math_ops.logical_or(decoder_finished, finished)

                next_sequence_lengths = array_ops.where(math_ops.logical_not(finished), array_ops.fill(array_ops.shape(sequence_lengths), time + 1), sequence_lengths)

                nest.assert_same_structure(state, decoder_state)
                nest.assert_same_structure(outputs_ta, next_outputs)
                nest.assert_same_structure(inputs, next_inputs)

                # Zero out output values past finish
                emit            = next_outputs
                next_state          = decoder_state
                outputs_ta      = nest.map_structure(lambda ta, out: ta.write(time, out), outputs_ta, emit)


                return (time + 1, outputs_ta, next_state, next_inputs, next_finished, next_sequence_lengths)


            for i in range(5):      #change this
                res = body(initial_time, initial_outputs_ta, initial_state, self.inputs_placeholder, initial_finished, initial_sequence_lengths)

                initial_time, initial_outputs_ta, initial_state, _, initial_finished, initial_sequence_lengths = res

                # out              = nest.map_structure(lambda ta: ta.stack(), initial_outputs_ta)
                # pdb.set_trace()
                print(i)

            final_outputs_ta        = res[1]
            final_state             = res[2]
            final_sequence_lengths  = res[5]


            final_outputs              = nest.map_structure(lambda ta: ta.stack(), final_outputs_ta)
            final_outputs, final_state = decoder.finalize(final_outputs, final_state, final_sequence_lengths)

            if not output_time_major:
                final_outputs              = nest.map_structure(_transpose_batch_time, final_outputs)

        return final_outputs, final_state, final_sequence_lengths


    def evaluate(self, sess):
        sess.run(self.val_init)

        with open('custom_beam_search', 'w') as f:
            while True:
                try:
                    # pdb.set_trace()
                    out = self.dynamic_decode(self.inference_decoder)
                    dec_out, next_outputs, ta = sess.run([self.dec_pred_decode, self.next_outputs[0], self.ta])
                    pdb.set_trace()
                    self.pred2txt(dec_out[0], f)

                    #write code to evaluate BLEU score here

                except tf.errors.OutOfRangeError:
                    break

    def cal_dpp(inp):
        return inp

    def run_epoch(self, sess, epoch):
        # Training step
        sess.run(self.train_init)

        ep_loss = 0.0
        step = 0
        while True:
            try:
                _, loss = sess.run([self.train_op, self.loss])
                ep_loss += loss
                # pdb.set_trace()
                # break
                step += 1

            except tf.errors.OutOfRangeError:
                break

            if step % self.p.patience == 0:
                self.logger.info("{} epoch: {} step: {} step_loss: {:.4f}".format(self.p.name, epoch, step, loss))

        if ep_loss < self.min_loss:
            self.saver.save(sess=sess, save_path=self.save_path)
            self.min_loss = ep_loss

        try:
            self.log_db.update_one(
                {'_id': self.p.name},
                {'$set': {
                    'train_loss':               float(ep_loss),
                    'Params':           vars(self.p),
                }
            }, upsert=True)

        except Exception as e:
            print('\nMongo ERROR Exception Cause: {}'.format(e.args[0]))

        self.logger.info('E:{} {} tr_loss: {:.3f} '.format(epoch, self.p.name, ep_loss))
        # self.evaluate(sess, split='test')

    def fit(self, sess):
        # self.summ_writer   = tf.summary.FileWriter("tf_board/Paraphraser/" + self.p.name, sess.graph)
        self.saver     = tf.train.Saver()
        save_dir       = 'checkpoints/' + self.p.name + '/'
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        self.save_path     = os.path.join(save_dir, 'best_val_loss')

        self.best_val_loss  = 0.0
        self.best_test_acc = 0.0

        if self.p.restore:
            self.saver.restore(sess, self.save_path)

        if self.p.mode == 'train':
            for epoch in range(self.p.max_epochs):
                self.run_epoch(sess, epoch)
        else:
            self.evaluate(sess)

        # self.evaluate(sess, split='test')

        # self.logger.info('Best Validation: {}, Best Test: {}, Best overall test {}'.format(self.best_val_acc, self.best_test_acc, self.ov_best_test_acc))


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