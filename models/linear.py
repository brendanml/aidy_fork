import tensorflow as tf


class Linear(tf.keras.models.Sequential):
    def __init__(self, num_alleles, feature_size, inner_act='relu', final_act='sigmoid'):
        super(Linear, self).__init__([
            tf.keras.layers.Input(shape=(feature_size,)),
            tf.keras.layers.Dense(256, activation=inner_act),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation=inner_act),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_alleles, activation=final_act)
        ])
        