# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Definition of a basic seq2seq model
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

from seq2seq.graph_utils import templatemethod
from seq2seq.models.model_base import ModelBase
from seq2seq.models.attention_seq2seq import AttentionSeq2Seq


class Image2Seq(AttentionSeq2Seq):
  """A model that encodes an image and produces a sequence
  of tokens.
  """

  def __init__(self, params, mode, name="image_seq2seq"):
    super(Image2Seq, self).__init__(params, mode, name)
    self.params["source.reverse"] = False
    self.params["embedding.share"] = False

  @staticmethod
  def default_params():
    params = ModelBase.default_params()
    params.update({
        "attention.class": "AttentionLayerBahdanau",
        "attention.params": {
            "num_units": 128
        },
        "bridge.class": "seq2seq.models.bridges.ZeroBridge",
        "bridge.params": {},
        "encoder.class": "seq2seq.encoders.InceptionV3Encoder",
        "encoder.params": {},  # Arbitrary parameters for the encoder
        "decoder.class": "seq2seq.decoders.AttentionDecoder",
        "decoder.params": {},  # Arbitrary parameters for the decoder
        "target.max_seq_len": 50,
        "embedding.dim": 100,
        "inference.beam_search.beam_width": 0,
        "inference.beam_search.length_penalty_weight": 0.0,
        "inference.beam_search.choose_successors_fn": "choose_top_k",
        "vocab_target": "",
    })
    return params

  @templatemethod("encode")
  def encode(self, features, _labels):
    encoder_fn = self.encoder_class(self.params["encoder.params"], self.mode)
    return encoder_fn(features["image"])

  def batch_size(self, features, _labels):
    return tf.shape(features["image"])[0]
