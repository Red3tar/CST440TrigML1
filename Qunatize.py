import pandas as pd
import numpy as np
import tensorflow as tf

df = pd.read_csv("testdata.csv")
x_vals = df["x"].values.astype("float32")

def rep_data():
    for v in x_vals:
        yield [np.array([[v]], dtype=np.float32)]


model = tf.keras.models.load_model("trig_model.keras")


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_data
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = converter.inference_output_type = tf.int8


tflite_model = converter.convert()
open("trigModel.tflite", "wb").write(tflite_model)