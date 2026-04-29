import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

MAX_LEN = 128

train = pd.read_csv("dataset/train.csv")
valid = pd.read_csv("dataset/valid.csv")
test = pd.read_csv("dataset/test.csv")

train = pd.concat([train, valid]).reset_index(drop=True)

classes = sorted(train["intent"].unique().tolist())

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = TFBertModel.from_pretrained("bert-base-uncased")

def encode_texts(texts):
    input_ids = []
    attention_masks = []

    for text in texts:
        enc = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="np"
        )

        input_ids.append(enc["input_ids"].flatten())
        attention_masks.append(enc["attention_mask"].flatten())

    return np.array(input_ids), np.array(attention_masks)

train_ids, train_masks = encode_texts(train["text"].tolist())
train_labels = np.array([classes.index(x) for x in train["intent"]])

input_ids = tf.keras.layers.Input(shape=(MAX_LEN,), dtype=tf.int32)
attention_mask = tf.keras.layers.Input(shape=(MAX_LEN,), dtype=tf.int32)

bert_output = bert_model(input_ids, attention_mask=attention_mask).last_hidden_state
cls_token = bert_output[:, 0, :]

x = tf.keras.layers.Dense(256, activation="relu")(cls_token)
x = tf.keras.layers.Dropout(0.3)(x)

output = tf.keras.layers.Dense(len(classes), activation="softmax")(x)

model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    [train_ids, train_masks],
    train_labels,
    validation_split=0.1,
    epochs=1,
    batch_size=8
)

model.save("intent_model.h5")

print("Model saved as intent_model.h5")