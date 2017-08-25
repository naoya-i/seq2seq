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
Sequence to Sequence model with attention
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pydoc import locate

import tensorflow as tf

from seq2seq import losses as seq2seq_losses
from seq2seq import decoders
from seq2seq.models.basic_seq2seq import BasicSeq2Seq


class AttentionSeq2Seq(BasicSeq2Seq):
  """Sequence2Sequence model with attention mechanism.

  Args:
    source_vocab_info: An instance of `VocabInfo`
      for the source vocabulary
    target_vocab_info: An instance of `VocabInfo`
      for the target vocabulary
    params: A dictionary of hyperparameters
  """

  def __init__(self, params, mode, name="att_seq2seq"):
    super(AttentionSeq2Seq, self).__init__(params, mode, name)

  def _preprocess(self, features, labels):
    features, labels = super(AttentionSeq2Seq, self)._preprocess(features, labels)

    if None != labels and "target_copysv" in labels:
        labels["target_copysv"] = labels["target_copysv"][:, :self.params[
            "target.max_seq_len"]]

    return features, labels

  @staticmethod
  def default_params():
    params = BasicSeq2Seq.default_params().copy()
    params.update({
        "attention.class": "AttentionLayerBahdanau",
        "attention.params": {}, # Arbitrary attention layer parameters
        "bridge.class": "seq2seq.models.bridges.ZeroBridge",
        "encoder.class": "seq2seq.encoders.BidirectionalRNNEncoder",
        "encoder.params": {},  # Arbitrary parameters for the encoder
        "decoder.class": "seq2seq.decoders.AttentionDecoder",
        "decoder.params": {}  # Arbitrary parameters for the decoder
    })
    return params

  def _create_decoder(self, encoder_output, features, _labels):
    attention_class = locate(self.params["attention.class"]) or \
      getattr(decoders.attention, self.params["attention.class"])
    attention_layer = attention_class(
        params=self.params["attention.params"], mode=self.mode)

    # If the input sequence is reversed we also need to reverse
    # the attention scores.
    reverse_scores_lengths = None
    if self.params["source.reverse"]:
      reverse_scores_lengths = features["source_len"]
      if self.use_beam_search:
        reverse_scores_lengths = tf.tile(
            input=reverse_scores_lengths,
            multiples=[self.params["inference.beam_search.beam_width"]])

    target_ids = None
    if self.params["inference.use_target_ids"]:
      target_ids = _labels["target_ids"]

    return self.decoder_class(
        params=self.params["decoder.params"],
        mode=self.mode,
        vocab_size=self.target_vocab_info.total_size,
        attention_values=encoder_output.attention_values,
        attention_values_length=encoder_output.attention_values_length,
        attention_keys=encoder_output.outputs,
        attention_fn=attention_layer,
        reverse_scores_lengths=reverse_scores_lengths,
        target_ids=target_ids,
        )

  def compute_loss(self, decoder_output, _features, labels):
    """Computes the loss for this model.

    Returns a tuple `(losses, loss)`, where `losses` are the per-batch
    losses and loss is a single scalar tensor to minimize.
    """
    #pylint: disable=R0201
    # Calculate loss per example-timestep of shape [B, T]
    losses = seq2seq_losses.cross_entropy_sequence_loss(
        logits=decoder_output.logits[:, :, :],
        targets=tf.transpose(labels["target_ids"][:, 1:], [1, 0]),
        sequence_length=labels["target_len"] - 1)

    # Calculate the average log perplexity
    loss = tf.reduce_sum(losses) / tf.to_float(
        tf.reduce_sum(labels["target_len"] - 1))

    if "target_copysv" in labels:
        lmd = 0.7
        targetsT = tf.transpose(labels["target_copysv"][:, 1:], [1, 0])

        masked_logits = tf.boolean_mask(decoder_output.attention_unscores[:, :, :], tf.not_equal(targetsT, -1))
        masked_copysv = tf.boolean_mask(targetsT, tf.not_equal(targetsT, -1))

        losses_copy = tf.nn.sparse_softmax_cross_entropy_with_logits(
             logits=masked_logits,
             labels=masked_copysv,
             )
        losses_copysw = seq2seq_losses.cross_entropy_sequence_loss(
             logits=decoder_output.copy_switch_prob[:, :, :],
             targets=tf.to_int32(tf.not_equal(tf.transpose(labels["target_copysv"][:, 1:], [1, 0]), -1)),
             sequence_length=labels["target_len"] - 1,
             )

        seq_loss = loss
        tf.summary.scalar("seq_loss", loss)

        copy_loss = tf.reduce_sum(losses_copy) / tf.to_float(tf.reduce_sum(labels["target_len"]))
        copysw_loss = tf.reduce_sum(losses_copysw) / tf.to_float(tf.reduce_sum(labels["target_len"]))

        tf.summary.scalar("copy_loss", copy_loss)
        tf.summary.scalar("copysw_loss", copysw_loss)
        loss = lmd*seq_loss + (1-lmd)*(copy_loss + copysw_loss)


    return losses, loss
