import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

data = pd.read_csv('20220419XFI5ecoV/Tensorflow2.0 Transformer模型中英翻译/data/cmn.txt', sep='\t')
enlish = data.iloc[:, 0]
chines = data.iloc[:, 1]

tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus((' '.join(e) for e in enlish), target_vocab_size=2**13)
tokenizer_cn = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus((' '.join(e) for e in chines), target_vocab_size=2**13)

cn = [tokenizer_cn.encode(line) for line in chines]
en= [tokenizer_en.encode(' '.join(line))for line in enlish]

en_ = []
for i in en :
    en_.append([tokenizer_en.vocab_size]+list(i)+[tokenizer_en.vocab_size+1])
cn_ = []
for i in cn :
    cn_.append([tokenizer_cn.vocab_size]+list(i)+[tokenizer_cn.vocab_size+1])

en_text=pad_sequences(
    en_, maxlen=40, dtype='int32', padding='post',
    value=0.0)
cn_text=pad_sequences(
    cn_, maxlen=40, dtype='int32', padding='post',
    value=0.0)

train_dataset= tf.data.Dataset.from_tensor_slices((en_text,cn_text))

train_dataset = train_dataset.shuffle(buffer_size=200).batch(64)

de_batch, en_batch = next(iter(train_dataset))






