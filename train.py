import sys

import keras
import keras_nlp
import pandas as pd

NUM_CLASSES = 17

preprocessor = keras_nlp.models.BertPreprocessor.from_preset("bert_base_en")

df = pd.read_json(sys.argv[1], lines=True)
df = df[
    df["text"]
    .apply(lambda t: t.get("questionType"))
    .isin(
        [
            "Opinion",
            "Discussion",
            "Advantages and Disadvantages",
            "Problem and Solution",
        ]
    )
]
scores = keras.utils.to_categorical(
    df["text"].apply(lambda t: t["band"]) * 2 - 2, NUM_CLASSES
)
questions = preprocessor(
    df["text"].apply(lambda t: t["question"]).replace("\r\n", "\n")
)
essays = preprocessor(df["text"].apply(lambda t: t["text"]).replace("\r\n", "\n"))

question = keras_nlp.models.BertBackbone.from_preset("bert_base_en")
question.trainable = False
for layer in question.layers:
    layer._name = f"question_{layer.name}"

essay = keras_nlp.models.BertBackbone.from_preset("bert_base_en")
essay.trainable = False
for layer in essay.layers:
    layer._name = f"essay_{layer.name}"

pooled = keras.layers.Concatenate()(
    [question(question.input)["pooled_output"], essay(essay.input)["pooled_output"]]
)
pooled = keras.layers.Dropout(0.1)(pooled)
outputs = keras.layers.Dense(
    NUM_CLASSES,
    activation="softmax",
    name="logits",
)(pooled)
model = keras.Model(inputs=[question.input, essay.input], outputs=outputs)
model.summary()
# keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)

model.compile(
    loss=keras.losses.CategoricalCrossentropy(),
    optimizer="adam",
    metrics=[keras.metrics.CategoricalAccuracy()],
)

model.fit((questions, essays), scores, epochs=10, batch_size=32)
