import os
import csv
import collections
import sys
from examples.bert.utils import data_utils, model_utils, tokenization
import importlib
import tensorflow as tf
import texar as tx 
from examples.bert import config_classifier as config_downstream
from texar.utils import transformer_utils
from examples.transformer.utils import data_utils, utils
from examples.transformer.bleu_tool import bleu_wrapper


######## Model architecture 
# Transformer encoder: BERT pretrained model 
# Transformer decoder: Transformer decoder stack with 6 layer 
# and each layer will have 8 multi heads 

#decoder config

dcoder_config = {
    'dim': 768,
    'num_blocks': 6,
    'multihead_attention': {
        'num_heads': 8,
        'output_dim': 768
        # See documentation for more optional hyperparameters
    },
    'position_embedder_hparams': {
        'dim': 768
    },
    'initializer': {
        'type': 'variance_scaling_initializer',
        'kwargs': {
            'scale': 1.0,
            'mode': 'fan_avg',
            'distribution': 'uniform',
        },
    },
    'poswise_feedforward': tx.modules.default_transformer_poswise_net_hparams(
        output_dim=768)
}

loss_label_confidence = 0.9

random_seed = 1234
beam_width = 5
alpha = 0.6
hidden_dim = 768
ma_iterations = 150000
opt = {
    'optimizer': {
        'type': 'AdamOptimizer',
        'kwargs': {
            'beta1': 0.9,
            'beta2': 0.997,
            'epsilon': 1e-9
        }
    }
}


lr = {
    'learning_rate_schedule': 'constant.linear_warmup.rsqrt_decay.rsqrt_depth',
    'lr_constant': 2 * (hidden_dim ** -0.5),
    'static_lr': 1e-3,
    'warmup_steps': 2000,
}

bos_token_id =101  #"[CLS]"
eos_token_id = 102 #"[SEP]"

model_dir= "./models"
run_mode= "train_and_evaluate"
batch_size = 32
test_batch_size = 32

max_train_epoch = 20
display_steps = 100
eval_steps = 1000

max_decoding_length = 128

max_seq_length_src = 64
max_seq_length_tgt = 128

bert_pretrain_dir = 'bert_pretrained_models/uncased_L-12_H-768_A-12'

## Extracting precomputed feature vectors from BERT 

class InputExample():
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence.
                For single sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second
                sequence. Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
                specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.src_txt = text_a
        self.tgt_txt = text_b
        
class InputFeatures():
    """A single set of features of data."""

    def __init__(self, src_input_ids,src_input_mask,src_segment_ids,tgt_input_ids,tgt_input_mask,tgt_labels):
        self.src_input_ids = src_input_ids
        self.src_input_mask = src_input_mask
        self.src_segment_ids = src_segment_ids
        self.tgt_input_ids = tgt_input_ids
        self.tgt_input_mask = tgt_input_mask 
        self.tgt_labels = tgt_labels

class template_data_processor(DataProcessor):

  def get_train_examples(self, data_dir):
    return self._create_examples(
        self._read_csv(os.path.join(data_dir, "train_data.tsv")), "train")
  def get_dev_examples(self, data_dir):
    return self._create_examples(
        self._read_csv(os.path.join(data_dir, "dev_data.tsv")), "dev")

  def get_test_examples(self, data_dir):
    return self._create_examples(
        self._read_csv(os.path.join(data_dir, "test_data.tsv")), "test")

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      text_a = tokenization.convert_to_unicode(line[2])
      text_b = tokenization.convert_to_unicode(line[1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b,))
    return examples
	
  def _read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter=",")
      lines = []
      for line in reader:
        lines.append(line)
      return lines 
 
## 
def file_based_convert_examples_to_features(
        examples, max_seq_length_src,max_seq_length_tgt,tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
          
        feature = convert_single_example(ex_index, example,
                                         max_seq_length_src,max_seq_length_tgt,tokenizer)

        def create_int_feature(values):
            return tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(values)))

        features = collections.OrderedDict()
        features["src_input_ids"] = create_int_feature(feature.src_input_ids)
        features["src_input_mask"] = create_int_feature(feature.src_input_mask)
        features["src_segment_ids"] = create_int_feature(feature.src_segment_ids)

        features["tgt_input_ids"] = create_int_feature(feature.tgt_input_ids)
        features["tgt_input_mask"] = create_int_feature(feature.tgt_input_mask)
        features['tgt_labels'] = create_int_feature(feature.tgt_labels)
        
        
        
        #print(feature.tgt_labels)
        

        tf_example = tf.train.Example(
            features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())

def convert_single_example(ex_index, example, max_seq_length_src,max_seq_length_tgt,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    """
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    """
    tokens_a = tokenizer.tokenize(example.src_txt)
    tokens_b = tokenizer.tokenize(example.tgt_txt)


    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
	
	# Sequences larger than max_seq_length will be truncated 
    if len(tokens_a) > max_seq_length_src - 2:
            tokens_a = tokens_a[0:(max_seq_length_src - 2)]
    
    if len(tokens_b) > max_seq_length_tgt - 2:
            tokens_b = tokens_b[0:(max_seq_length_tgt - 2)]

    
    tokens_src = []
    segment_ids_src = []
    tokens_src.append("[CLS]") ## "[CLS]" will be appended and will be first token of the input
    segment_ids_src.append(0)  ## All the tokens to the input will have "0" segment id
    for token in tokens_a:
        tokens_src.append(token)
        segment_ids_src.append(0)
    tokens_src.append("[SEP]")  
    segment_ids_src.append(0)

    tokens_tgt = []            
    segment_ids_tgt = []
    tokens_tgt.append("[CLS]")
    #segment_ids_tgt.append(0)
    for token in tokens_b:
		tokens_tgt.append(token)
        #segment_ids_tgt.append(0)
    tokens_tgt.append("[SEP]")
    #segment_ids_tgt.append(0)

    input_ids_src = tokenizer.convert_tokens_to_ids(tokens_src)
    input_ids_tgt = tokenizer.convert_tokens_to_ids(tokens_tgt)
    
    #Adding begiining and end token
	input_ids_tgt_copy = input_ids_tgt
    input_ids_tgt = input_ids_tgt[:-1]              
    input_mask_src = [1] * len(input_ids_src)
    input_mask_tgt = [1] * len(input_ids_tgt)
	labels_tgt = input_ids_tgt_copy[1:]
  
    labels_tgt.append(0)
    
    while len(input_ids_src) < max_seq_length_src:
        input_ids_src.append(0)
        input_mask_src.append(0)
        segment_ids_src.append(0)

    while len(input_ids_tgt) < max_seq_length_tgt:
        input_ids_tgt.append(0)
		input_mask_tgt.append(0)
        segment_ids_tgt.append(0)
        labels_tgt.append(0)

    feature = InputFeatures( src_input_ids=input_ids_src,src_input_mask=input_mask_src,src_segment_ids=segment_ids_src,
        tgt_input_ids=input_ids_tgt,tgt_input_mask=input_mask_tgt,tgt_labels=labels_tgt)

    
    return feature
	

def file_based_input_fn_builder(input_file, max_seq_length_src,max_seq_length_tgt, is_training,
                                drop_remainder, is_distributed=False):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "src_input_ids": tf.FixedLenFeature([max_seq_length_src], tf.int64),
        "src_input_mask": tf.FixedLenFeature([max_seq_length_src], tf.int64),
        "src_segment_ids": tf.FixedLenFeature([max_seq_length_src], tf.int64),
        "tgt_input_ids": tf.FixedLenFeature([max_seq_length_tgt], tf.int64),
        "tgt_input_mask": tf.FixedLenFeature([max_seq_length_tgt], tf.int64),
        "tgt_labels" : tf.FixedLenFeature([max_seq_length_tgt], tf.int64),
        
        
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)
        print(example)
        print(example.keys())

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
		    if is_distributed:
                import horovod.tensorflow as hvd
                tf.logging.info('distributed mode is enabled.'
                                'size:{} rank:{}'.format(hvd.size(), hvd.rank()))
                # https://github.com/uber/horovod/issues/223
                d = d.shard(hvd.size(), hvd.rank())

                d = d.repeat()
                d = d.shuffle(buffer_size=100)
                d = d.apply(
                    tf.contrib.data.map_and_batch(
                        lambda record: _decode_record(record, name_to_features),
                        batch_size=batch_size//hvd.size(),
                        drop_remainder=drop_remainder))
            else:
                tf.logging.info('distributed mode is not enabled.')
                d = d.repeat()
                d = d.shuffle(buffer_size=100)
                d = d.apply(
                    tf.contrib.data.map_and_batch(
                        lambda record: _decode_record(record, name_to_features),
                        batch_size=batch_size,
                        drop_remainder=drop_remainder))

        else:
            d = d.apply(
                tf.contrib.data.map_and_batch(
                    lambda record: _decode_record(record, name_to_features),
                    batch_size=batch_size,
                    drop_remainder=drop_remainder))

        return d
    return input_fn
	
def get_dataset(processor,
                tokenizer,
                data_dir,
                max_seq_length_src,
                max_seq_length_tgt,
                batch_size,
                mode,
                output_dir,
                is_distributed=False):
    """
    Args:
        processor: Data Preprocessor, must have get_lables,
            get_train/dev/test/examples methods defined.
        tokenizer: The Sentence Tokenizer. Generally should be
            SentencePiece Model.
        data_dir: The input data directory.
        max_seq_length: Max sequence length.
        batch_size: mini-batch size.
        model: `train`, `eval` or `test`.
        output_dir: The directory to save the TFRecords in.
    """
    if mode == 'train':
        train_examples = processor.get_train_examples(data_dir)
        train_file = os.path.join(output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, max_seq_length_src,max_seq_length_tgt,
             tokenizer, train_file)
        dataset = file_based_input_fn_builder(
            input_file=train_file,
            max_seq_length_src=max_seq_length_src,
            max_seq_length_tgt =max_seq_length_tgt,
            is_training=True,
            drop_remainder=True,
            is_distributed=is_distributed)({'batch_size': batch_size})
    elif mode == 'eval':
		eval_examples = processor.get_dev_examples(data_dir)
        eval_file = os.path.join(output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, max_seq_length_src,max_seq_length_tgt,
            tokenizer, eval_file)
        dataset = file_based_input_fn_builder(
            input_file=eval_file,
            max_seq_length_src=max_seq_length_src,
            max_seq_length_tgt =max_seq_length_tgt,
            is_training=False,
            drop_remainder=True,
            is_distributed=is_distributed)({'batch_size': batch_size})
    elif mode == 'test':
      
        test_examples = processor.get_test_examples(data_dir)
        test_file = os.path.join(output_dir, "predict.tf_record")
        
        file_based_convert_examples_to_features(
            test_examples, max_seq_length_src,max_seq_length_tgt,
            tokenizer, test_file)
        dataset = file_based_input_fn_builder(
            input_file=test_file,
            max_seq_length_src=max_seq_length_src,
            max_seq_length_tgt =max_seq_length_tgt,
            is_training=False,
            drop_remainder=True,
            is_distributed=is_distributed)({'batch_size': batch_size})
    return dataset
	
bert_config = model_utils.transform_bert_to_texar_config(
            os.path.join(bert_pretrain_dir, 'bert_config.json'))



tokenizer = tokenization.FullTokenizer(
        vocab_file=os.path.join(bert_pretrain_dir, 'vocab.txt'),
        do_lower_case=True)

vocab_size = len(tokenizer.vocab)
processor = CNNDailymail()
train_dataset = get_dataset(processor,tokenizer,"./",max_seq_length_src,max_seq_length_tgt,batch_size,'train',"./")
eval_dataset = get_dataset(processor,tokenizer,"./",max_seq_length_src,max_seq_length_tgt,batch_size,'eval',"./")
test_dataset = get_dataset(processor,tokenizer,"./",max_seq_length_src,max_seq_length_tgt,batch_size,'test',"./")

#inputs to the model
src_input_ids = tf.placeholder(tf.int64, shape=(None, None))
src_segment_ids = tf.placeholder(tf.int64, shape=(None, None))
tgt_input_ids = tf.placeholder(tf.int64, shape=(None, None))
tgt_segment_ids = tf.placeholder(tf.int64, shape=(None, None))

batch_size = tf.shape(src_input_ids)[0]

src_input_length = tf.reduce_sum(1 - tf.to_int32(tf.equal(src_input_ids, 0)),
                             axis=1)
tgt_input_length = tf.reduce_sum(1 - tf.to_int32(tf.equal(src_input_ids, 0)),
                             axis=1)

labels = tf.placeholder(tf.int64, shape=(None, None))
is_target = tf.to_float(tf.not_equal(labels, 0))


global_step = tf.Variable(0, dtype=tf.int64, trainable=False)
learning_rate = tf.placeholder(tf.float64, shape=(), name='lr')

#create the iterator 
iterator = tx.data.FeedableDataIterator({
        'train': train_dataset, 'eval': eval_dataset, 'test': test_dataset})
batch = iterator.get_next()

#encoder Bert model
print("Intializing the Bert Encoder Graph")
with tf.variable_scope('bert'):
        embedder = tx.modules.WordEmbedder(
            vocab_size=bert_config.vocab_size,
            hparams=bert_config.embed)
        word_embeds = embedder(src_input_ids)

        # Creates segment embeddings for each type of tokens.
        segment_embedder = tx.modules.WordEmbedder(
            vocab_size=bert_config.type_vocab_size,
            hparams=bert_config.segment_embed)
        segment_embeds = segment_embedder(src_segment_ids)

        input_embeds = word_embeds + segment_embeds

        # The BERT model (a TransformerEncoder)
        encoder = tx.modules.TransformerEncoder(hparams=bert_config.encoder)
        encoder_output = encoder(input_embeds, src_input_length)
        
        # Builds layers for downstream classification, which is also initialized
        # with BERT pre-trained checkpoint.
        with tf.variable_scope("pooler"):
            # Uses the projection of the 1st-step hidden vector of BERT output
            # as the representation of the sentence
            bert_sent_hidden = tf.squeeze(encoder_output[:, 0:1, :], axis=1)
            bert_sent_output = tf.layers.dense(
                bert_sent_hidden, config_downstream.hidden_dim,
                activation=tf.tanh)
            output = tf.layers.dropout(
                bert_sent_output, rate=0.1, training=tx.global_mode_train())

print("loading the bert pretrained weights")
init_checkpoint = os.path.join(bert_pretrain_dir, 'bert_model.ckpt')
model_utils.init_bert_checkpoint(init_checkpoint)

#decoder part and mle losss
tgt_embedding = tf.concat(
    [tf.zeros(shape=[1, embedder.dim]), embedder.embedding[1:, :]], axis=0)

decoder = tx.modules.TransformerDecoder(embedding=tgt_embedding,
                             hparams=dcoder_config)
# For training
outputs = decoder(
    memory=encoder_output,
    memory_sequence_length=src_input_length,
    inputs=embedder(tgt_input_ids),
    sequence_length=tgt_input_length,
    decoding_strategy='train_greedy',
    mode=tf.estimator.ModeKeys.TRAIN
)

mle_loss = transformer_utils.smoothing_cross_entropy(
        outputs.logits, labels, vocab_size, loss_label_confidence)
mle_loss = tf.reduce_sum(mle_loss * is_target) / tf.reduce_sum(is_target)

train_op = tx.core.get_train_op(
        mle_loss,
        learning_rate=learning_rate,
        global_step=global_step,
        hparams=opt)

tf.summary.scalar('lr', learning_rate)
tf.summary.scalar('mle_loss', mle_loss)
summary_merged = tf.summary.merge_all()

#prediction 
start_tokens = tf.fill([tx.utils.get_batch_size(src_input_ids)],
                       bos_token_id)
predictions = decoder(
    memory=encoder_output,
    memory_sequence_length=src_input_length,
    decoding_strategy='infer_greedy',
    beam_width=beam_width,
    alpha=alpha,
    start_tokens=start_tokens,
    end_token=eos_token_id,
    max_decoding_length=400,
    mode=tf.estimator.ModeKeys.PREDICT
)
if beam_width <= 1:
    inferred_ids = predictions[0].sample_id
else:
    # Uses the best sample by beam search
    inferred_ids = predictions['sample_id'][:, :, 0]


saver = tf.train.Saver(max_to_keep=5)
best_results = {'score': 0, 'epoch': -1}

def infer_single_eample(input_text, output_text, tokenizer):
   src_input = input_text
   tgt_input = output_text
   ex_id = 1000
   example = InputExample(guid=ex_id,text_a=src_input,text_b=tgt_input):
   features = convert_single_example(ex_id,example,max_seq_length_src,max_seq_length_tgt,tokenzier)
   id2w = tokenizer.inv_vocab
   references,hypothesis = [],[]
   
   feed_dict={
   src_input_ids:np.array(features['src}
   return src_input
   
