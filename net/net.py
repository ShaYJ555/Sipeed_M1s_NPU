import tensorflow as tf


Backbone = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(240,240,3),
                                                        alpha=1.0,
                                                        include_top=False,
                                                        weights=None)
inputs = tf.keras.layers.Input(shape=(240,240,3))
x = Backbone(inputs)
x = tf.keras.layers.GlobalAvgPool2D()(x)
x = tf.keras.layers.Dense(1000,activation='relu')(x)
outputs = tf.keras.layers.Dense(68*2)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

if __name__ == '__main__':
    model.summary()

    