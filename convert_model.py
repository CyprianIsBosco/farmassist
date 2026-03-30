import tensorflow as tf

print("Loading model...")
model = tf.keras.models.load_model("plant_disease.h5")

print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("plant_disease.tflite", "wb") as f:
    f.write(tflite_model)

print("Done! plant_disease.tflite created successfully")