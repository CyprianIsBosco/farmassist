import tensorflow as tf
import numpy as np

print("Loading model...")
model = tf.keras.models.load_model("plant_disease.h5")
print(f"Model input shape: {model.input_shape}")
print(f"Model output shape: {model.output_shape}")

print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float32]
tflite_model = converter.convert()

with open("plant_disease.tflite", "wb") as f:
    f.write(tflite_model)

print(f"Done! File size: {len(tflite_model)} bytes")

print("Verifying the converted model...")
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(f"Input shape: {input_details[0]['shape']}")
print(f"Output shape: {output_details[0]['shape']}")
print("Verification successful!")