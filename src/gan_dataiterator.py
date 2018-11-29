# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions for downloading and reading MNIST data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip

import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes

class DataSet(object):
    def __init__(self,
                 orig_sentences,
                 orig_hidden_state,
                 para_sentences,
                 para_hidden_state,
                 fake_data=False,
                 one_hot=False,
                 dtype=dtypes.float32,
                 reshape=False):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        """
        self._num_examples = orig_sentences.shape[0]

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        self._orig_sentences = orig_sentences
        self._orig_hidden_state = orig_hidden_state
        self._para_sentences = para_sentences
        self._para_hidden_state = para_hidden_state
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def orig_sentences(self):
        return self._orig_sentences

    @property
    def orig_hidden_state(self):
        return self._orig_hidden_state

    @property
    def para_sentences(self):
        return self._orig_sentences

    @property
    def para_hidden_state(self):
        return self._orig_hidden_state

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm0)
            self._orig_sentences = self.orig_sentences[perm0]
            self._orig_hidden_state = self.orig_hidden_state[perm0]
            self._para_sentences = self.para_sentences[perm0]
            self._para_hidden_state = self.para_hidden_state[perm0]

        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            orig_sentences_rest_part = self._orig_sentences[start:self._num_examples]
            orig_hidden_state_rest_part = self._orig_hidden_state[start:self._num_examples]
            para_sentences_rest_part = self._para_sentences[start:self._num_examples]
            para_hidden_state_rest_part = self._para_hidden_state[start:self._num_examples]

            # Shuffle the data
            if shuffle:
                perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._orig_sentences = self.orig_sentences[perm]
            self._orig_hidden_state = self.orig_hidden_state[perm]
            self._para_sentences = self.para_sentences[perm]
            self._para_hidden_state = self.para_hidden_state[perm]

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            orig_sentences_new_part = self._orig_sentences[start:end]
            orig_hidden_state_new_part = self._orig_hidden_state[start:end]
            para_sentences_new_part = self._para_sentences[start:end]
            para_hidden_state_new_part = self._para_hidden_state[start:end]

            return numpy.concatenate((orig_sentences_rest_part, orig_sentences_new_part), axis=0) , \
                numpy.concatenate((orig_hidden_state_rest_part, orig_hidden_state_new_part), axis=0) , \
                numpy.concatenate((para_sentences_rest_part, para_sentences_new_part), axis=0) , \
                numpy.concatenate((para_hidden_state_rest_part, para_hidden_state_new_part), axis=0) , \

        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._orig_sentences[start:end], self._orig_hidden_state[start:end], \
                self._para_sentences[start:end], self._para_hidden_state[start:end]
