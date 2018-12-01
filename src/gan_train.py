import math

import numpy as np
import tensorflow as tf
import os, glob, gzip, time, pdb
import logging

from datetime import datetime
from collections import OrderedDict
from collections import namedtuple

#from src.commons.utils import *
from helper import *
from gan_dataloader import DataLoader

class GANModel:

    def __init__(self, config, logger):
        OPTIMIZERS = {
                        "adam":     tf.train.AdamOptimizer,
                        "adadelta": tf.train.AdadeltaOptimizer,
                        "rmsprop":  tf.train.RMSPropOptimizer,
                        "sgd":      tf.train.GradientDescentOptimizer
                    }

        # Fill this
        self.batch_size     = config.gan_batch_size
        self.lambda_wgangp  = config.lambda_wgangp
        self.max_grad_norm  = config.gan_max_grad_norm
        self.logger         = logger

        self.generator_layer_units = list(map(int, config.generator_layer_units.split(',')))

        self.generator_output_size = config.hidden_size

        self.gen_optimizer      = OPTIMIZERS[config.gen_optimizer]
        self.gen_learning_rate  = config.gen_learning_rate
        self.z_dim              = config.z_dim

        # ROTH GAN Specs
        self.gan_reg_type = config.gan_reg_type

        # Discriminator specifications
        self.discriminator_layer_units = list(map(int, config.discriminator_layer_units.split(',')))
        self.disc_optimizer            = OPTIMIZERS[config.disc_optimizer]
        self.disc_learning_rate        = config.disc_learning_rate

        # Global step counter
        self.global_step            = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step      = tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op   = tf.assign(self.global_epoch_step, self.global_epoch_step+1)

        self.initialize_placeholder()
        self.build_model()

    def initialize_placeholder(self):
        with tf.name_scope("placeholder"):
            self.z           = tf.placeholder(tf.float32, (None, self.z_dim), name="z")
            self.hidden_orig = tf.placeholder(tf.float32, (None, self.generator_output_size), name="hidden_orig")
            self.hidden_para = tf.placeholder(tf.float32, (None, self.generator_output_size), name="hidden_para")
            self.keep_prob   = tf.placeholder(tf.float32, [], name="keep_prob")

            if self.gan_reg_type == 'rothgan':
               self.gamma_plh = tf.placeholder(tf.float32, shape=(), name='gamma')

    def build_model(self):
        self.logger.info("Building Model...")

        self.gen_output     = self.generator(self.z)
        self.diff_para_orig = self.hidden_para - self.hidden_orig

        self.disc_real      = self.discriminator(self.diff_para_orig)
        self.disc_fake      = self.discriminator(self.gen_output)

        self._gen_optimizer()
        self._disc_optimizer()

        self.summary = tf.summary.merge_all()

    def generator(self, z_noise):
        generator_input = z_noise
        self.logger.info("Building generator...")

        with tf.variable_scope("encoder/generator"):        #adding enoder to restore it from paraphraser
            layer_out = []
            layer_out.append(tf.contrib.layers.fully_connected(generator_input, self.generator_layer_units[0]))

            for i, layer in enumerate(self.generator_layer_units[1:]):
                layer_out.append(tf.contrib.layers.fully_connected(layer_out[i], layer))

            perturbation = tf.contrib.layers.fully_connected(layer_out[len(layer_out)-1], self.generator_output_size, activation_fn=None)

        return perturbation

    def discriminator(self, discriminator_input):
        self.logger.info("Building discriminator...")

        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
            layer_out = []
            layer_out.append(tf.contrib.layers.fully_connected(discriminator_input, self.discriminator_layer_units[0]))

            for i, layer in enumerate(self.discriminator_layer_units[1:]):
                layer_out.append(tf.contrib.layers.fully_connected(layer_out[i], layer))

            discriminator_output = tf.contrib.layers.fully_connected(layer_out[len(layer_out)-1], 1, activation_fn=None) #no activation function

        return discriminator_output

    def _gen_loss(self):
        with tf.name_scope("generator_loss"):
            if self.gan_reg_type == 'wgangp':                       # Wasserstein GAN with gradient penalty
                self.gen_loss = -tf.reduce_mean(self.disc_fake)
            else:                                                   # This is for ROTH GAN
                self.gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.disc_fake, labels=tf.ones_like(self.disc_fake)))

        tf.summary.scalar("gen_loss", self.gen_loss)

    def _disc_loss(self):
        with tf.name_scope("discriminator_loss"):
            if self.gan_reg_type == 'wgangp':
                self.disc_loss = tf.reduce_mean(self.disc_fake) - tf.reduce_mean(self.disc_real)

                alpha           = tf.random_uniform(shape=[self.batch_size, 1], minval=0., maxval=1.)
                differences     = tf.subtract(self.gen_output, self.diff_para_orig)
                interpolates    = self.diff_para_orig + (alpha*differences)
                gradients       = tf.gradients(self.discriminator(interpolates), [interpolates])[0]
                slopes          = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
                gradient_penalty = tf.reduce_mean((slopes -1.)**2)

                self.disc_loss += self.lambda_wgangp*gradient_penalty

            else:
                self.disc_loss           =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.disc_real, labels=tf.ones_like(self.disc_real)))
                self.disc_loss          += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.disc_fake, labels=tf.zeros_like(self.disc_fake)))
                self.d1                  = tf.nn.sigmoid(self.disc_real)
                self.d2                  = tf.nn.sigmoid(self.disc_fake)
                self.grad_d1_logits      = tf.gradients(self.disc_real, self.diff_para_orig)[0]
                self.grad_d2_logits      = tf.gradients(self.disc_fake, self.gen_output)[0]
                self.grad_d1_logits_norm = tf.norm(self.grad_d1_logits, axis=1)
                self.grad_d2_logits_norm = tf.norm(self.grad_d2_logits, axis=1)

                reg_D1 = tf.multiply(tf.square(1.0-self.d1), tf.square(self.grad_d1_logits_norm))
                reg_D2 = tf.multiply(tf.square(self.d2), tf.square(self.grad_d2_logits_norm))

                self.disc_reg   = tf.reduce_mean(reg_D1 + reg_D2)
                self.disc_loss += (self.gamma_plh/2.0)*self.disc_reg

        tf.summary.scalar("disc_loss", self.disc_loss)

    def _gen_optimizer(self):
        self._gen_loss()
        self.logger.info("Setting optimizer")

        with tf.name_scope("generator_optimizer"):
            generator_trainable_params = tf.trainable_variables("encoder/generator")
            self.gen_opt               = self.gen_optimizer(learning_rate=self.gen_learning_rate)
            generator_grads            = tf.gradients(self.gen_loss, generator_trainable_params)
            clip_grads, _              = tf.clip_by_global_norm(generator_grads, self.max_grad_norm)

            self.gen_updates = self.gen_opt.apply_gradients(zip(clip_grads, generator_trainable_params),global_step=self.global_step)

    def _disc_optimizer(self):
        self._disc_loss()
        self.logger.info("Setting discriminator optimizer")

        with tf.name_scope("discriminator_optimizer"):
            discriminator_trainable_params = tf.trainable_variables("discriminator")
            self.disc_opt                  = self.disc_optimizer(learning_rate=self.disc_learning_rate)

            discriminator_grads            = tf.gradients(self.disc_loss, discriminator_trainable_params)
            clip_grads, _                  = tf.clip_by_global_norm(discriminator_grads, self.max_grad_norm)

            self.disc_updates = self.disc_opt.apply_gradients(zip(clip_grads, discriminator_trainable_params),global_step=self.global_step)

    def save(self, sess, path, var_list=None, global_step=None):
        saver     = tf.train.Saver(var_list)
        save_path = saver.save(sess, save_path=path, global_step=global_step)
        self.logger.info("Model saved at {}".format(save_path))

    def restore(self, sess, path, var_list=None):
        saver = tf.train.Saver(var_list)
        saver.restore(sess, save_path=path)
        self.logger.info("Model restored from {}".format(path))

    def train_discriminator(self, sess, z_noise, hidden_orig, hidden_para, gamma=0.1, keep_prob = 0.0):
        inputs = {
                "z_noise":     z_noise,
                "hidden_orig": hidden_orig,
                "hidden_para": hidden_para,
                "keep_prob":   keep_prob
                }

        if self.gan_reg_type == 'rothgan':
            inputs['gamma'] = gamma

        input_feed  = self.check_feeds(inputs, is_generator=False)
        output_feed = [self.disc_loss, self.disc_updates, self.summary]

        outputs = sess.run(output_feed, input_feed)
        return outputs

    def eval_discriminator(self, sess, z_noise, hidden_orig, hidden_para, gamma=0.1, keep_prob = 0.0):
        inputs = {
                "z_noise" : z_noise,
                "hidden_orig" : hidden_orig,
                "hidden_para" : hidden_para,
                "keep_prob" : keep_prob
                }


        if self.gan_reg_type == 'rothgan':
            inputs['gamma'] = gamma

        input_feed  = self.check_feeds(inputs, is_generator=False)
        output_feed = [self.disc_loss, self.summary]
        outputs     = sess.run(output_feed, input_feed)

        return outputs

    def predict_discriminator(self, sess, z_noise, hidden_orig, hidden_para, gamma=0.1, keep_prob = 0.0):
        inputs = {
                "z_noise" : z_noise,
                "hidden_orig" : hidden_orig,
                "hidden_para" : hidden_para,
                "keep_prob" : keep_prob
                }

        if self.gan_reg_type == 'rothgan':
            inputs['gamma'] = gamma

        input_feed  = self.check_feeds(inputs)
        output_feed = [self.disc_real, self.disc_fake]

        outputs = sess.run(output_feed, input_feed)
        return outputs

    def train_generator(self, sess, z_noise, hidden_orig, hidden_para, gamma=0.1, keep_prob = 0.0):
        inputs = {
                "z_noise" : z_noise,
                "hidden_orig" : hidden_orig,
                "hidden_para" : hidden_para,
                "keep_prob" : keep_prob
                }

        if self.gan_reg_type == 'rothgan':
            inputs['gamma'] = gamma

        #print(inputs)
        input_feed  = self.check_feeds(inputs, is_generator=True)
        output_feed = [self.gen_loss, self.gen_updates, self.summary]

        outputs = sess.run(output_feed, input_feed)
        return outputs

    def eval_generator(self, sess, z_noise, hidden_orig, hidden_para, gamma=0.1, keep_prob = 0.0):
        inputs = {
                "z_noise" : z_noise,
                "hidden_orig" : hidden_orig,
                "hidden_para": hidden_para,
                "keep_prob" : keep_prob
                }

        if self.gan_reg_type == 'rothgan':
            inputs['gamma'] = gamma

        input_feed  = self.check_feeds(inputs, is_generator=True)
        output_feed = [self.gen_loss, self.summary]
        outputs     = sess.run(output_feed, input_feed)

        return outputs

    def predict_generator(self, sess, z_noise, hidden_orig, hidden_para, gamma=0.1, keep_prob = 0.0):
        inputs = {
                "z_noise" : z_noise,
                "hidden_orig" : hidden_orig,
                "hidden_para" : hidden_para,
                "keep_prob" : keep_prob
                }

        if self.gan_reg_type == 'rothgan':
            inputs['gamma'] = gamma

        input_feed = self.check_feeds(inputs, is_generator=True)
        output_feed = [self.gen_output]
        # Keep prob dropout
        outputs = sess.run(output_feed, input_feed)
        return outputs

    def check_feeds(self, inputs, is_generator):
        # Sanity check for input data like shape, etc.

        if is_generator:
            input_feed = self.check_feeds_generator(inputs)
        else:
            input_feed = self.check_feeds_discriminator(inputs)

        return input_feed

    def check_feeds_generator(self, inputs):
        input_feed = {}

        if inputs["z_noise"].shape[0] != inputs["hidden_orig"].shape[0]:
            self.logger.error("Shape mismatch between z_noise and hidden_orig")
            raise ValueError("Shape mismatch between z_noise and hidden_orig. found {} and {}".format(inputs['z_noise'].shape[0], inputs['hidden_orig'].shape[0]))

        input_feed[self.z.name] = inputs["z_noise"]
        input_feed[self.hidden_orig.name] = inputs["hidden_orig"]
        input_feed[self.hidden_para.name] = inputs["hidden_para"]
        input_feed[self.keep_prob.name] = inputs["keep_prob"]

        if self.gan_reg_type == 'rothgan':
            input_feed[self.gamma_plh.name] = inputs['gamma']

        return input_feed

    def check_feeds_discriminator(self, inputs):
        input_feed = {}

        if inputs["z_noise"].shape[0] != inputs["hidden_orig"].shape[0]:
            self.logger.error("Shape mismatch between z_noise and hidden_orig")
            raise ValueError("Shape mismatch between z_noise and hidden_orig")
        if inputs["z_noise"].shape[0] != inputs["hidden_para"].shape[0]:
            self.logger.error("Shape mismatch between z_noise and hidden_para")
            raise ValueError("Shape mismatch between z_noise and hidden_para")
        if inputs["hidden_orig"].shape[1] != inputs["hidden_para"].shape[1] or inputs["hidden_orig"].shape[0] != inputs["hidden_para"].shape[0]:
            self.logger.error("Shape mismatch between z_noise and hidden_para")
            raise ValueError("Shape mismatch between z_noise and hidden_para")

        input_feed[self.z.name] = inputs["z_noise"]
        input_feed[self.hidden_orig.name] = inputs["hidden_orig"]
        input_feed[self.hidden_para.name] = inputs["hidden_para"]
        input_feed[self.keep_prob.name] = inputs["keep_prob"]

        if self.gan_reg_type == 'rothgan':
            input_feed[self.gamma_plh.name] = inputs['gamma']

        return input_feed

def create_model(session, config, logger, gan_model_path):
    model = GANModel(config, logger)
    ckpt  = tf.train.get_checkpoint_state(os.path.join(gan_model_path,'checkpoint'))

    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        logger.info("Reloading model parameters")
        model.restore(session, ckpt.model_checkpoint_path)
    else:
        if not os.path.exists(gan_model_path): os.makedirs(gan_model_path)

        logger.info("Created new model parameters")
        session.run(tf.global_variables_initializer())

    return model

def sample_z(batch_size, z_dim):
    z_batch = np.random.normal(size=(batch_size, z_dim)).astype('float32')
    return z_batch

def run_gan_model(args, gpu_config):
    logger = get_logger(args.name, args.log_dir, args.config_dir)
    logger.info("Loading Data... from {}".format(args.data))

    train_data = DataLoader(args.data, "json", args.metadata.split(','), "train").dataset
    logger.info("Data loading completed")

    saver_path     = os.path.join('ganmodel_checkpoints', args.name)

    with tf.Session(config=gpu_config) as sess:
        ganmodel     = create_model(sess, args, logger, saver_path)
        train_writer = tf.summary.FileWriter(os.path.join(saver_path, "train"), sess.graph)
        num_batches  = int(train_data.num_examples/args.gan_batch_size)
        num_batches  = 10

        logger.info("TRAINING STARTED...")
        for i in range(args.gan_max_epochs):
            j = 0
            while j < num_batches:
                j+= 1
                outputs                 = train_data.next_batch(args.gan_batch_size)
                batch_orig_sentence     = outputs[0]
                batch_orig_hidden_state = outputs[1]
                batch_para_sentence     = outputs[2]
                batch_para_hidden_state = outputs[3]

                z_batch              = sample_z(args.gan_batch_size, args.z_dim)
                gen_loss, _, summary = ganmodel.train_generator(sess, z_batch, batch_orig_hidden_state, batch_para_hidden_state)
                gstep                = ganmodel.global_step.eval()

                train_writer.add_summary(summary, gstep)
                logger.info("EPOCH : {}, ITERATION : {}, GENERATOR LOSS : {}".format(i, gstep, gen_loss))

                for k in range(args.discriminator_iterations):
                    outputs = train_data.next_batch(args.gan_batch_size)

                    j += 1
                    if j >= num_batches: break

                    batch__sentence         = outputs[0]
                    batch_orig_hidden_state = outputs[1]
                    batch_para_sentence     = outputs[2]
                    batch_para_hidden_state = outputs[3]

                    z_batch               = sample_z(args.gan_batch_size, args.z_dim)
                    disc_loss, _, summary = ganmodel.train_discriminator(sess, z_batch, batch_orig_hidden_state, batch_para_hidden_state)

                    gstep = ganmodel.global_step.eval()
                    logger.info("EPOCH : {}, ITERATION : {}, DISCRIMINATOR LOSS : {}".format(i, gstep, disc_loss))
                    train_writer.add_summary(summary, gstep)

            ganmodel.save(sess, saver_path, None, i)

        with open(os.path.join(saver_path, 'params'), 'w') as f:
            params = vars(args)
            params = {k: v for k,v in params.items() if k != 'dtype'}
            json.dump(params, f)

if __name__== "__main__":

    parser = argparse.ArgumentParser(description='GAN model')

    parser.add_argument('-data',        dest="data",       default='gan_data.json',        help='path to hidden states')
    parser.add_argument('-seq2seq',     dest="seq2seq",    default='',                     help='path to hidden states')
    parser.add_argument('-gpu',         dest="gpu",        default='0',                    help='GPU to use')
    parser.add_argument('-name',        dest="name",       default='test',                 help='Name of the run')

    parser.add_argument('-metadata',      dest="metadata",      default='train,inp_sent,inp_hidden,tar_sent,tar_hidden', help='keys of json file')

    parser.add_argument('-annealing',         dest='annealing',         default=False,                  help='whether to anneal or not')
    parser.add_argument("-decay_factor",      dest='decay_factor',      default=0.01,   type=float,     help="exponential annealing decay rate")
    parser.add_argument("-lambda_wgangp",     dest='lambda_wgangp',     default=5,      type=int,       help="Lambda parameter of WGAN gradient penalty")
    parser.add_argument("-gamma",             dest='gamma',             default=0.1,    type=float,     help="noise variance for regularizer")
    parser.add_argument("-gan_max_grad_norm", dest='gan_max_grad_norm', default=1.0,    type=float,     help="Clip gradients to this norm")
    parser.add_argument("-gan_reg_type",      dest='gan_reg_type',      default="rothgan",              help="type of regularizer to use: wgangp / rothgan")
    parser.add_argument("-gan_max_epochs",    dest='gan_max_epochs',    default=50,     type=int,       help="Number of epochs to be run")
    parser.add_argument('-batch_size',        dest="gan_batch_size",    default=32,     type=int,       help='batch size to use')

    parser.add_argument("-mean_noise",            dest='mean_noise',            default=0,      type=int,                    help="mean of the noise z to feed to the generator")
    parser.add_argument("-sigma_noise",           dest='sigma_noise',           default=1,      type=int,                   help="variance of the noise z to feed to the generator")
    parser.add_argument("-gen_learning_rate",     dest='gen_learning_rate',     default=0.0002, type=float,                    help="Learning rate for generator")
    parser.add_argument("-gen_optimizer",         dest='gen_optimizer',         default="adam",                     help="Select optimizer from 'adam', 'adadelta', 'rmsprop', 'sgd'")
    parser.add_argument("-generator_layer_units", dest='generator_layer_units', default="256, 384",                      help="Number of units in each layer in the generator")
    parser.add_argument("-z_dim",                 dest='z_dim',                 default=64,      type=int,                    help="Dimension of the noise parameter z to feed to the generator")

    # Discriminator configurations
    parser.add_argument("-disc_learning_rate",        dest='disc_learning_rate',        default=0.0002, type=float,                 help="Learning rate for discriminator")
    parser.add_argument("-disc_optimizer",            dest='disc_optimizer',            default="adam",                 help="Select optimizer from 'adam', 'adadelta', 'rmsprop', 'sgd'")
    parser.add_argument("-discriminator_layer_units", dest='discriminator_layer_units', default="256",                  help="Number of units in each layer in the discriminator")
    parser.add_argument("-discriminator_iterations",  dest='discriminator_iterations',  default=5, type=int,                     help="Number of iterations to train the discriminator before training the generator")

    parser.add_argument('-seed',        dest="seed",       default=1234,        type=int,       help='Seed for randomization')
    parser.add_argument('-logdir',      dest="log_dir",    default='/scratchd/home/shikhar/gcn_word_embed/src/log/',       help='Log directory')
    parser.add_argument('-config',      dest="config_dir", default='../config/',       help='Config directory')
    parser.add_argument('-hidden_size', dest="hidden_size",         default=512,        type=int,       help='Hidden dimensions of the enc/dec')

    args = parser.parse_args()
    args.data = 'checkpoints/{}/{}'.format(args.seq2seq, args.data)

    tf.set_random_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    set_gpu(args.gpu)

    config                          = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.operation_timeout_in_ms  = 60000

    run_gan_model(args, config)
    print('Model Trained Successfully!!')
