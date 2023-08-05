import keras
import tensorflow as tf

from preprocessor import preprocessor

model = keras.models.load_model("saved_model")


def predict(question, essay):
    return (
        tf.argmax(
            model.predict((preprocessor([question]), preprocessor([essay]))), axis=1
        )[0].numpy()
        + 2
    ) / 2
