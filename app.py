import pandas as pd
import numpy as np
import tensorflow as tf
import gradio as gr
from transformers import BertTokenizer, TFBertModel

MAX_LEN = 128

# =====================
# DATA + LABELS
# =====================
train = pd.read_csv("dataset/train.csv")
valid = pd.read_csv("dataset/valid.csv")

train = pd.concat([train, valid]).reset_index(drop=True)

classes = sorted(train["intent"].unique().tolist())

# =====================
# TOKENIZER
# =====================
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# =====================
# MODEL CLASS
# =====================
class IntentModel(tf.keras.Model):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = TFBertModel.from_pretrained("bert-base-uncased")
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.dense = tf.keras.layers.Dense(256, activation="relu")
        self.classifier = tf.keras.layers.Dense(
            num_classes,
            activation="softmax"
        )

    def call(self, inputs):
        input_ids, attention_mask = inputs

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_token = outputs.last_hidden_state[:, 0, :]

        x = self.dropout(cls_token)
        x = self.dense(x)

        return self.classifier(x)

# =====================
# LOAD MODEL
# =====================
model = IntentModel(len(classes))

# Build once
dummy_ids = tf.zeros((1, MAX_LEN), dtype=tf.int32)
dummy_mask = tf.zeros((1, MAX_LEN), dtype=tf.int32)

model((dummy_ids, dummy_mask))

# Load trained weights
model.load_weights("intent_model.weights.h5")

# =====================
# PREDICT
# =====================
def predict_intent(text):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="np"
    )

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    preds = model.predict(
        (input_ids, attention_mask),
        verbose=0
    )

    predicted_class_index = np.argmax(preds, axis=1)[0]

    return classes[predicted_class_index]

# =====================
# ORIGINAL STYLE UI
# =====================
iface = gr.Interface(
    fn=predict_intent,
    inputs="text",
    outputs="text",
    title="Intent Detection",
    description="Enter a phrase to predict the intent."
)

iface.launch()