import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout


# Authored by Google (or MT TODO: check)
def _get_last_layer_units_and_activation(num_classes):
    """Gets the # units and activation function for the last network layer.

    # Arguments
        num_classes: int, number of classes.

    # Returns
        units, activation values.
    """
    if num_classes == 2:
        activation = 'sigmoid'
        units = 1
    else:
        activation = 'softmax'
        units = num_classes
    return units, activation


# # Authored by Google (or MT TODO: check)
def mlp_model(layers, units, dropout_rate, input_shape, num_classes):
    """Creates an instance of a multi-layer perceptron model.

    # Arguments
        layers: int, number of `Dense` layers in the model.
        units: int, output dimension of the layers.
        dropout_rate: float, percentage of input to drop at Dropout layers.
        input_shape: tuple, shape of input to the model.
        num_classes: int, number of output classes.

    # Returns
        An MLP model instance.
    """
    op_units, op_activation = _get_last_layer_units_and_activation(num_classes)
    model = models.Sequential()
    model.add(Dropout(rate=dropout_rate, input_shape=input_shape))
    for _ in range(layers - 1):
        model.add(Dense(units=units, activation='relu'))
        model.add(Dropout(rate=dropout_rate))
    model.add(Dense(units=op_units, activation=op_activation))
    return model


# Authored by Google (or MT TODO: check)
def train_ngram_model(llms,
                      x_train_dict,
                      x_val_dict,
                      x_test_dict,
                      train_labels_dict,
                      val_labels_dict,
                      num_classes,
                      results_path,
                      learning_rate=3 * 1e-4,
                      epochs=100,
                      batch_size=128,
                      layers=2,
                      units=64,
                      dropout_rate=0.2):
    """Trains n-gram model on the given dataset.

    # Arguments
        data: tuples of training and test texts and labels.
        learning_rate: float, learning rate for training model.
        epochs: int, number of epochs.
        batch_size: int, number of samples per batch.
        layers: int, number of `Dense` layers in the model.
        units: int, output dimension of Dense layers in the model.
        dropout_rate: float: percentage of input to drop at Dropout layers.

    # Raises
        ValueError: If validation data has label values which were not seen
            in the training data.
    """
    # Initialise return variables
    history_acc_dict = {}
    history_val_acc_dict = {}
    y_pred_dict = {}

    for llm in llms:
        x_train = x_train_dict[llm]
        x_val = x_val_dict[llm]
        x_test = x_test_dict[llm]
        train_labels = train_labels_dict[llm]
        val_labels = val_labels_dict[llm]

        # Create model instance.
        model = mlp_model(layers=layers,
                          units=units,
                          dropout_rate=dropout_rate,
                          input_shape=x_train.shape[1:],
                          num_classes=num_classes)

        # Compile model with learning parameters.
        if num_classes == 2:
            loss = 'binary_crossentropy'
        else:
            loss = 'sparse_categorical_crossentropy'
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

        # Create callback for early stopping on validation loss. If the loss does
        # not decrease in two consecutive tries, stop training.
        callbacks = [tf.keras.callbacks.EarlyStopping(
            monitor='val_acc', patience=10)]  # val_loss

        # Train and validate model.
        history = model.fit(
            x_train,
            train_labels,
            epochs=epochs,
            # callbacks=callbacks, # UNCOMMENT TO GET EARLY STOPPING
            validation_data=(x_val, val_labels),
            verbose=2,  # Logs once per epoch.
            batch_size=batch_size
        )
        # print(4)
        # Print results.
        history = history.history
        # print('Training accuracy: {acc}, loss: {loss}'.format( #Validation accuracy; loss
        #        acc=history['acc'][-1], loss=history['loss'][-1])) #val_acc, val_loss

        # Save model.
        model.save(f'{results_path}/{llm}_model.h5')  # Eventually pass a name to this function so that it saves one
        # model per LLM
        y_pred = model.predict(x_test).ravel()

        # Add the llm-specific arrays to the return dicts
        history_acc_dict[llm] = history['acc']
        history_val_acc_dict[llm] = history['val_acc']
        y_pred_dict[llm] = y_pred

    return history_acc_dict, history_val_acc_dict, y_pred_dict


if __name__ == "__main__":
    # Preprocess the data
    from preprocessor import Preprocessor
    kwargs_preprocessor = {
        "llms": ["gpt3.04", "gpt3.5", "gpt3.041", "gpt3.042", "gpt3.043", "gpt4_1106_cot", "gpt4_1106", "llama007"],
        "path_to_dataset_outputs": "datasets/cladder/outputs",
        "ngram_range": (1, 2),
        "top_k_ngrams": 20_000,
        "token_mode": "word",
        "min_document_frequency": 2,
        "num_classes": 2,
        "split_ratio": 0.2,
    }
    preprocessor = Preprocessor(**kwargs_preprocessor)
    preprocessor.preprocess_dataset()

    # Train the MLPs
    kwargs_trainer = {
        "llms": ["gpt3.04", "gpt3.5", "gpt3.041", "gpt3.042", "gpt3.043", "gpt4_1106_cot", "gpt4_1106", "llama007"],
        "x_train_dict": preprocessor.x_train_dict,
        "x_val_dict": preprocessor.x_val_dict,
        "x_test_dict": preprocessor.x_test_dict,
        "train_labels_dict": preprocessor.train_labels_dict,
        "val_labels_dict": preprocessor.val_labels_dict,
        "num_classes": 2,
        "results_path": "results"
    }
    train_ngram_model(**kwargs_trainer)

    print("Exit ok")