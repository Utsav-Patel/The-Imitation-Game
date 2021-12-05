import tensorflow as tf


def create_model_project1_agent_10x10():
    model = tf.keras.models.Sequential([
        # tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(2048, activation='relu', input_shape=(100,)),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        # tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax'),
        # tf.keras.layers.Softmax()
    ])

    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
                                    name='Adam')

    model.compile(optimizer=adam,
                  loss=loss_fn,
                  metrics=['accuracy'])

    return model


def create_model_project1_agent_20x20():
    model = tf.keras.models.Sequential([
        # tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(4096, activation='relu', input_shape=(400,)),
        tf.keras.layers.Dense(2048, activation='relu'),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax'),
    ])

    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
                                    name='Adam')

    model.compile(optimizer=adam,
                  loss=loss_fn,
                  metrics=['accuracy'])

    return model
