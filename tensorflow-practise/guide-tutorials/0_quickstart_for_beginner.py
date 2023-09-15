import tensorflow as tf
import logging

logging.getLogger().setLevel(logging.INFO)


if __name__ == '__main__':
    logging.info("Tensorflow version: %s" % str(tf.__version__))

    # 载入MNIST数据集
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # 搭建模型
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    # 训练并验证模型
    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test, verbose=2)

    # if you want your model to return a probability, you can wrap the trained model, and attach the softmax to it
    probability_model = tf.keras.Sequential([
        model,
        tf.keras.layers.Softmax()
    ])
    print(probability_model(x_test[:5]))
