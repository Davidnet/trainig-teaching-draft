import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path

(ds_train, ds_test), ds_info = tfds.load(
    "kmnist",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,
    data_dir="./data",
    with_info=True,
)


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255.0, label


ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)


ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10),
    ]
)

model_version = 1
tb_callback = tf.keras.callbacks.TensorBoard(
    f"./logs/logs-exp{model_version}", update_freq=1
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model.summary()
model.fit(ds_train, epochs=6, validation_data=ds_test, callbacks=[tb_callback])

model_dir = Path("./models")
(model_dir / "saved").mkdir(exist_ok=True)
(model_dir / "export").mkdir(exist_ok=True)

model.save(f"./models/saved/model-exp{model_version}.keras")
model.export(f"./models/export/model-exp{model_version}")
