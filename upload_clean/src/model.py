"""Model builder utilities.

Provides a small CNN for demo and a helper to build a transfer-learning
backbone if TensorFlow and Keras applications are available.
"""

def build_small_cnn(input_shape=(128, 128, 3), num_classes=3):
    try:
        from tensorflow.keras import layers, models
    except Exception as e:
        raise RuntimeError("TensorFlow/Keras is required to build the model: " + str(e))

    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_model(name: str = "small", input_shape=(128, 128, 3), num_classes: int = 3):
    """Factory: 'small' for demo or 'resnet50' for transfer learning (if available)."""
    if name == "small":
        return build_small_cnn(input_shape=input_shape, num_classes=num_classes)
    elif name == "resnet50":
        try:
            from tensorflow.keras import layers, models
            from tensorflow.keras.applications import ResNet50
        except Exception as e:
            raise RuntimeError("TensorFlow/Keras + applications required for resnet50: " + str(e))

        base = ResNet50(weights=None, include_top=False, input_shape=input_shape)
        x = base.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation="relu")(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        model = models.Model(inputs=base.input, outputs=outputs)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model
    else:
        raise ValueError(f"Unknown model name: {name}")
