import pandas as pd
import numpy as np
import tensorflow as tf

# user_NN=tf.keras.models.Sequential({
#     tf.keras.layers.Dense(256,activation='relu'),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(32)
# })
#
# item_NN=tf.keras.models.Sequential({
#     tf.keras.layers.Dense(256,activation='relu'),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(32)
# })
#
# input_user=tf.keras.layers.Input(shape=(user_features))
# vu=user_NN(input_user)
# vu=tf.linalg.12_normalize(vu,axis=1)
#
# input_item=tf.keras.layers.Input(shape=(user_features))
# vm=user_NN(input_user)
# vm=tf.linalg.12_normalize(vu,axis=1)
#
# output=tf.keras.layers.Dot(axes=1)([vu,vm])
#
# model=Model([input_user,input_item],output)
#
# cost_fn=tf.keras.losses.MeanSquaredError()

# Example input: 10 features (after preprocessing)
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


user_input = keras.Input(shape=(10,), name="user_features")
u = layers.Dense(64, activation="relu")(user_input)
user_embedding = layers.Dense(32, activation=None, name="user_embedding")(u)
user_model = keras.Model(user_input, user_embedding, name="user_model")


video_input = keras.Input(shape=(10,), name="video_features")
v = layers.Dense(64, activation="relu")(video_input)
video_embedding = layers.Dense(32, activation=None, name="video_embedding")(v)
video_model = keras.Model(video_input, video_embedding, name="video_model")


dot_product = layers.Dot(axes=1)([user_embedding, video_embedding])
score = layers.Activation("sigmoid")(dot_product)

recommendation_model = keras.Model(
    inputs=[user_input, video_input],
    outputs=score,
    name="recommendation_model"
)
recommendation_model.compile(optimizer="adam", loss="binary_crossentropy")

#dummy data
user_features = np.random.rand(100, 10)   # 100 users × 10 features
video_features = np.random.rand(100, 10)  # 100 videos × 10 features
labels = np.random.randint(0, 2, size=(100, 1))

# Train
recommendation_model.fit(
    [user_features, video_features],
    labels,
    epochs=5,
    batch_size=16
)


user_emb = user_model.predict(user_features[:1])# for one user
video_embs = video_model.predict(video_features)

scores = tf.linalg.matmul(user_emb, video_embs, transpose_b=True)
top_indices = tf.argsort(scores[0], direction="DESCENDING")[:5]
print("Top recommended videos:", top_indices.numpy())
