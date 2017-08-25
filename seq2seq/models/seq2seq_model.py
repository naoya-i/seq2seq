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
"""Base class for models"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections

import numpy as np
import tensorflow as tf

from tensorflow import gfile
from seq2seq import graph_utils
from seq2seq import losses as seq2seq_losses
from tensorflow.contrib.seq2seq.python.ops.helper import _transpose_batch_time
from seq2seq.data import vocab
from seq2seq.graph_utils import templatemethod
from seq2seq.decoders.beam_search_decoder import BeamSearchDecoder
from seq2seq.inference import beam_search
from seq2seq.models.model_base import ModelBase, _flatten_dict


class Seq2SeqModel(ModelBase):
  """Base class for seq2seq models with embeddings
  """

  def __init__(self, params, mode, name):
    super(Seq2SeqModel, self).__init__(params, mode, name)

    self.source_vocab_info = None
    if "vocab_source" in self.params and self.params["vocab_source"]:
      self.source_vocab_info = vocab.get_vocab_info(self.params["vocab_source"])

    self.target_vocab_info = None
    if "vocab_target" in self.params and self.params["vocab_target"]:
      self.target_vocab_info = vocab.get_vocab_info(self.params["vocab_target"])

    self.source_emb_init = tf.random_uniform_initializer(
        -self.params["embedding.init_scale"],
        self.params["embedding.init_scale"])
    self.target_emb_init = tf.random_uniform_initializer(
      -self.params["embedding.init_scale"],
      self.params["embedding.init_scale"])

    if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
        if self.params["embedding.pretrained.source"] != "":
          self.source_emb_init = self._load_pretrained_emb(self.params["vocab_source"], self.params["embedding.pretrained.source"], self.params["embedding.pretrained_vocab.source"])

        if not self.params["embedding.share"]:
          if self.params["embedding.pretrained.target"] != "":
            self.target_emb_init = self._load_pretrained_emb(self.params["vocab_target"], self.params["embedding.pretrained.target"], self.params["embedding.pretrained_vocab.target"])


  @staticmethod
  def default_params():
    params = ModelBase.default_params()
    params.update({
        "source.max_seq_len": 50,
        "source.reverse": True,
        "target.max_seq_len": 50,
        "embedding.dim": 100,
        "embedding.init_scale": 0.04,
        "embedding.share": False,
        "embedding.trainable": True,
        "embedding.pretrained.source": "",
        "embedding.pretrained.target": "",
        "embedding.pretrained_vocab.source": "",
        "embedding.pretrained_vocab.target": "",
        "inference.beam_search.beam_width": 0,
        "inference.beam_search.length_penalty_weight": 0.0,
        "inference.beam_search.choose_successors_fn": "choose_top_k",
        "inference.use_target_ids": False,
        "optimizer.clip_embed_gradients": 0.1,
        "vocab_source": "",
        "vocab_target": "",
    })
    return params


  def _load_pretrained_emb(self, vocab_path, emb_path, emb_vocab_path):

    # Loading pre-trained embeddings
    with gfile.GFile(vocab_path) as file:
      vocab = list(line.split("\t")[0] for line in file)

    vocab_map = {}

    for i, w in enumerate(vocab):
      vocab_map[w] = i

    # Loading vectors
    tf.logging.info("Loading pretrained embeddings from {}...".format(emb_path))

    npemb = np.load(open(emb_path, "rb"))
    npemb_vocab = open(emb_vocab_path).read().splitlines()

    embeddings = list(2.0*(np.random.random(self.params["embedding.dim"])-0.5)*self.params["embedding.init_scale"] for i in range(len(vocab)+3))
    load_count = 0

    with gfile.GFile(emb_vocab_path) as file:
      for word, vec in zip(file, npemb):
        word = word.strip()

        if word in vocab_map:
          load_count += 1
          embeddings[vocab_map[word]] = vec

    tf.logging.info("Loaded {} pretrained vectors (vocab: {}) from {}".format(load_count, len(vocab_map), emb_path))

    return tf.constant_initializer(np.array(embeddings, dtype=np.float32), verify_shape=True)


  # def _clip_gradients(self, grads_and_vars):
  #   """In addition to standard gradient clipping, also clips embedding
  #   gradients to a specified value."""
  #   grads_and_vars = super(Seq2SeqModel, self)._clip_gradients(grads_and_vars)
  #
  #   clipped_gradients = []
  #   variables = []
  #   for gradient, variable in grads_and_vars:
  #     if "embedding" in variable.name:
  #       tmp = tf.clip_by_norm(
  #           gradient.values, self.params["optimizer.clip_embed_gradients"])
  #       gradient = tf.IndexedSlices(tmp, gradient.indices, gradient.dense_shape)
  #     clipped_gradients.append(gradient)
  #     variables.append(variable)
  #   return list(zip(clipped_gradients, variables))

  def _create_predictions(self, decoder_output, features, labels, losses=None):
    """Creates the dictionary of predictions that is returned by the model.
    """
    predictions = {}

    # Add features and, if available, labels to predictions
    predictions.update(_flatten_dict({"features": features}))
    if labels is not None:
      predictions.update(_flatten_dict({"labels": labels}))

    if losses is not None:
      predictions["losses"] = _transpose_batch_time(losses)

    # Decoders returns output in time-major form [T, B, ...]
    # Here we transpose everything back to batch-major for the user
    output_dict = collections.OrderedDict(
        zip(decoder_output._fields, decoder_output))
    decoder_output_flat = _flatten_dict(output_dict)
    decoder_output_flat = {
        k: _transpose_batch_time(v)
        for k, v in decoder_output_flat.items()
    }
    predictions.update(decoder_output_flat)

    # If we predict the ids also map them back into the vocab and process them
    if "predicted_ids" in predictions.keys():
      vocab_tables = graph_utils.get_dict_from_collection("vocab_tables")
      target_id_to_vocab = vocab_tables["target_id_to_vocab"]
      predicted_tokens = target_id_to_vocab.lookup(
          tf.to_int64(predictions["predicted_ids"]))
      # Raw predicted tokens
      predictions["predicted_tokens"] = predicted_tokens

    return predictions

  def batch_size(self, features, labels):
    """Returns the batch size of the curren batch based on the passed
    features.
    """
    return tf.shape(features["source_ids"])[0]

  @property
  @templatemethod("source_embedding")
  def source_embedding(self):
    """Returns the embedding used for the source sequence.
    """
    return tf.get_variable(
        name="W",
        shape=[self.source_vocab_info.total_size, self.params["embedding.dim"]],
        initializer=self.source_emb_init,
        trainable=self.params["embedding.trainable"])

  @property
  @templatemethod("target_embedding")
  def target_embedding(self):
    """Returns the embedding used for the target sequence.
    """
    if self.params["embedding.share"]:
      return self.source_embedding
    return tf.get_variable(
        name="W",
        shape=[self.target_vocab_info.total_size, self.params["embedding.dim"]],
        initializer=self.target_emb_init,
        trainable=self.params["embedding.trainable"])

  @templatemethod("encode")
  def encode(self, features, labels):
    """Encodes the inputs.
    """
    raise NotImplementedError()

  @templatemethod("decode")
  def decode(self, encoder_output, features, labels):
    """Runs decoding based on the encoder outputs.
    """
    raise NotImplementedError()

  def _get_beam_search_decoder(self, decoder):
    """Wraps a decoder into a Beam Search decoder.

    Args:
      decoder: The original decoder

    Returns:
      A BeamSearchDecoder with the same interfaces as the original decoder.
    """
    config = beam_search.BeamSearchConfig(
        beam_width=self.params["inference.beam_search.beam_width"],
        vocab_size=self.target_vocab_info.total_size,
        eos_token=self.target_vocab_info.special_vocab.SEQUENCE_END,
        length_penalty_weight=self.params[
            "inference.beam_search.length_penalty_weight"],
        choose_successors_fn=getattr(
            beam_search,
            self.params["inference.beam_search.choose_successors_fn"]))
    return BeamSearchDecoder(decoder=decoder, config=config)

  @property
  def use_beam_search(self):
    """Returns true iff the model should perform beam search.
    """
    return self.params["inference.beam_search.beam_width"] > 1

  def _preprocess(self, features, labels):
    """Model-specific preprocessing for features and labels:

    - Creates vocabulary lookup tables for source and target vocab
    - Converts tokens into vocabulary ids
    """

    # Create vocabulary lookup for source
    source_vocab_to_id, source_id_to_vocab, source_word_to_count, _ = \
      vocab.create_vocabulary_lookup_table(self.source_vocab_info.path)

    # Create vocabulary look for target
    target_vocab_to_id, target_id_to_vocab, target_word_to_count, _ = \
      vocab.create_vocabulary_lookup_table(self.target_vocab_info.path)

    # Add vocab tables to graph colection so that we can access them in
    # other places.
    graph_utils.add_dict_to_collection({
        "source_vocab_to_id": source_vocab_to_id,
        "source_id_to_vocab": source_id_to_vocab,
        "source_word_to_count": source_word_to_count,
        "target_vocab_to_id": target_vocab_to_id,
        "target_id_to_vocab": target_id_to_vocab,
        "target_word_to_count": target_word_to_count
    }, "vocab_tables")

    # Slice source to max_len
    if self.params["source.max_seq_len"] is not None:
      features["source_tokens"] = features["source_tokens"][:, :self.params[
          "source.max_seq_len"]]
      features["source_len"] = tf.minimum(features["source_len"],
                                          self.params["source.max_seq_len"])

    # Look up the source ids in the vocabulary
    features["source_ids"] = source_vocab_to_id.lookup(features[
        "source_tokens"])

    # Maybe reverse the source
    if self.params["source.reverse"] is True:
      features["source_ids"] = tf.reverse_sequence(
          input=features["source_ids"],
          seq_lengths=features["source_len"],
          seq_dim=1,
          batch_dim=0,
          name=None)

    features["source_len"] = tf.to_int32(features["source_len"])
    tf.summary.histogram("source_len", tf.to_float(features["source_len"]))

    if labels is None:
      return features, None

    labels = labels.copy()

    # Slices targets to max length
    if self.params["target.max_seq_len"] is not None:
      labels["target_tokens"] = labels["target_tokens"][:, :self.params[
          "target.max_seq_len"]]
      labels["target_len"] = tf.minimum(labels["target_len"],
                                        self.params["target.max_seq_len"])

    # Look up the target ids in the vocabulary
    labels["target_ids"] = target_vocab_to_id.lookup(labels["target_tokens"])

    labels["target_len"] = tf.to_int32(labels["target_len"])
    tf.summary.histogram("target_len", tf.to_float(labels["target_len"]))

    # Keep track of the number of processed tokens
    num_tokens = tf.reduce_sum(labels["target_len"])
    num_tokens += tf.reduce_sum(features["source_len"])
    token_counter_var = tf.Variable(0, "tokens_counter")
    total_tokens = tf.assign_add(token_counter_var, num_tokens)
    tf.summary.scalar("num_tokens", total_tokens)

    with tf.control_dependencies([total_tokens]):
      features["source_tokens"] = tf.identity(features["source_tokens"])

    # Add to graph collection for later use
    graph_utils.add_dict_to_collection(features, "features")
    if labels:
      graph_utils.add_dict_to_collection(labels, "labels")

    return features, labels

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

    return losses, loss

  def _build(self, features, labels, params):
    # Pre-process features and labels
    features, labels = self._preprocess(features, labels)

    encoder_output = self.encode(features, labels)
    decoder_output, _, = self.decode(encoder_output, features, labels)

    if self.mode == tf.contrib.learn.ModeKeys.INFER:
      predictions = self._create_predictions(
          decoder_output=decoder_output, features=features, labels=labels)
      loss = None
      train_op = None
    else:
      losses, loss = self.compute_loss(decoder_output, features, labels)

      train_op = None
      if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
        train_op = self._build_train_op(loss)

      predictions = self._create_predictions(
          decoder_output=decoder_output,
          features=features,
          labels=labels,
          losses=losses)

    # We add "useful" tensors to the graph collection so that we
    # can easly find them in our hooks/monitors.
    graph_utils.add_dict_to_collection(predictions, "predictions")

    return predictions, loss, train_op
