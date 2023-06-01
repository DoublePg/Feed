from tensorflow.keras.models import Model, Sequential, clone_model, load_model
from tensorflow.keras.layers import Input, Dense, add, concatenate, Conv2D, Dropout, \
    BatchNormalization, Flatten, MaxPooling2D, AveragePooling2D, Activation, Dropout, Reshape, \
    ZeroPadding2D, RepeatVector, multiply, MaxPool2D, ReLU, GlobalAvgPool2D, Add, Embedding, LSTM, Conv1D, MaxPooling1D, \
    GlobalMaxPooling1D, Softmax, ReLU
from keras.initializers import Constant
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf


def cnn1d_model(n_classes, num_words, embedding_dim, max_sequence_length, embedding_matrix, dropout=0.3):
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = Embedding(
        num_words,
        embedding_dim,
        embeddings_initializer=Constant(embedding_matrix),
        input_length=max_sequence_length,
        trainable=True
    )(sequence_input)

    cnn1 = Conv1D(256, 3, padding='same', strides=1, activation='relu', name="Conv1")(
        embedded_sequences)
    cnn1 = MaxPooling1D(pool_size=48)(cnn1)

    cnn2 = Conv1D(256, 4, padding='same', strides=1, activation='relu', name="Conv2")(embedded_sequences)
    cnn2 = MaxPooling1D(pool_size=47)(cnn2)

    cnn3 = Conv1D(256, 5, padding='same', strides=1, activation='relu', name="Conv3")(embedded_sequences)
    cnn3 = MaxPooling1D(pool_size=46)(cnn3)

    cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)

    flat = Flatten(name='embeddings')(cnn)
    drop = Dropout(dropout)(flat)
    x = Dense(n_classes, kernel_regularizer=tf.keras.regularizers.l2(1e-3), activation=None, use_bias=False, name='fc1')(
        drop)
    preds = Activation('softmax', name='pred')(x)

    model = Model(sequence_input, preds)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy']
    )
    return model


def mmoe_cnn1d_model(n_classes, num_words, embedding_dim, max_sequence_length, embedding_matrix, num_experts=3):
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = Embedding(
        num_words,
        embedding_dim,
        embeddings_initializer=Constant(embedding_matrix),
        input_length=max_sequence_length,
        trainable=True
    )(sequence_input)
    experts = []
    for i in range(num_experts):

        cnn1 = Conv1D(256, 3, padding='same', strides=1, activation='relu', name="Conv1_expert" + str(i))(
            embedded_sequences)
        cnn1 = MaxPooling1D(pool_size=48)(cnn1)

        cnn2 = Conv1D(256, 4, padding='same', strides=1, activation='relu', name="Conv2_expert" + str(i))(
            embedded_sequences)
        cnn2 = MaxPooling1D(pool_size=47)(cnn2)

        cnn3 = Conv1D(256, 5, padding='same', strides=1, activation='relu', name="Conv3_expert" + str(i))(
            embedded_sequences)
        cnn3 = MaxPooling1D(pool_size=46)(cnn3)

        cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)

        flat = Flatten()(cnn)
        drop = Dropout(0.3)(flat)
        experts.append(drop)

    input_flatten = Flatten()(embedded_sequences)

    # gate
    g = Dense(units=num_experts, activation='softmax', name='gate')(input_flatten)

    # attention
    att = Dense(units=200, activation='relu', use_bias=False, name='Att_hidden')(
        tf.stack(experts, axis=1))
    att = Dropout(0.5)(att)
    att = Dense(units=100, activation='relu', use_bias=False, name='Att_hidden2')(att)
    att = Dropout(0.5)(att)
    att = Dense(units=50, activation='relu', use_bias=False, name='Att_hidden3')(att)
    att = Dropout(0.5)(att)
    att = Dense(units=1, activation=None, use_bias=False, name='Att')(att)
    att = Dropout(0.5)(att)
    attention_probs = Softmax(name='Att_softmax')(tf.squeeze(att, axis=-1))
    g = multiply([attention_probs, g], name='weight')

    g = RepeatVector(experts[0].shape[-1])(g)
    Embeddings = []
    for i in range(num_experts):
        gi = multiply([g[:, :, i], experts[i]])
        Embeddings.append(gi)

    added_embedding = add(Embeddings, name='embeddings')
    t = Dense(n_classes, activation=None, use_bias=False,
              kernel_regularizer=tf.keras.regularizers.l2(1e-3), name='logit')(added_embedding)  # added_embedding
    pred = Activation("softmax", name='pred')(t)

    model = Model(sequence_input, pred)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy']
    )
    return model


def mmoe_cnn3layer_model(n_classes, n1=128, n2=192, n3=256, dropout_rate=0.2, input_shape=(28, 28),
                                num_experts=3):
    model_A, x = None, None
    x = Input(input_shape)
    if len(input_shape) == 2:
        input = Reshape((input_shape[0], input_shape[1], 1))(x)
    else:
        input = Reshape(input_shape)(x)

    experts = []
    for i in range(num_experts):
        y = Conv2D(filters=n1, kernel_size=(3, 3), strides=1, padding="same",
                   activation=None, name="Conv1_expert" + str(i))(input)
        y = BatchNormalization(name="BN1_expert" + str(i))(y)
        y = Activation("relu")(y)
        y = Dropout(dropout_rate)(y)
        y = AveragePooling2D(pool_size=(2, 2), strides=1, padding="same")(y)

        y = Conv2D(filters=n2, kernel_size=(2, 2), strides=2, padding="valid",
                   activation=None, name="Conv2_expert" + str(i))(y)
        y = BatchNormalization(name="BN2_expert" + str(i))(y)
        y = Activation("relu")(y)
        y = Dropout(dropout_rate)(y)
        y = AveragePooling2D(pool_size=(2, 2), strides=2, padding="valid")(y)

        y = Conv2D(filters=n3, kernel_size=(3, 3), strides=2, padding="valid",
                   activation=None, name="Conv3_expert" + str(i))(y)
        y = BatchNormalization(name="BN3_expert" + str(i))(y)
        y = Activation("relu")(y)
        y = Dropout(dropout_rate)(y)

        y = Flatten()(y)
        experts.append(y)

    input_flatten = Flatten()(input)
    g = Dense(units=num_experts, activation='softmax', name='gate')(input_flatten)

    # attention
    att = Dense(units=200, activation='relu', use_bias=False, name='Att_hidden')(
        tf.stack(experts, axis=1))
    att = Dropout(0.5)(att)
    att = Dense(units=100, activation='relu', use_bias=False, name='Att_hidden2')(att)
    att = Dropout(0.5)(att)
    att = Dense(units=50, activation='relu', use_bias=False, name='Att_hidden3')(att)
    att = Dropout(0.5)(att)
    att = Dense(units=1, activation=None, use_bias=False, name='Att')(att)
    att = Dropout(0.5)(att)
    attention_probs = Softmax(name='Att_softmax')(tf.squeeze(att, axis=-1))

    g = multiply([attention_probs, g], name='weight')
    g = RepeatVector(experts[0].shape[-1])(g)

    Added_Embedding = []
    for i in range(num_experts):
        gi = multiply([g[:, :, i], experts[i]])
        Added_Embedding.append(gi)

    embedding = add(Added_Embedding, name='embeddings')

    # tower
    t = Dense(units=n_classes, activation=None, use_bias=False,
              kernel_regularizer=tf.keras.regularizers.l2(1e-3), name='logit')(embedding)

    pred = Activation("softmax", name='pred')(t)

    model_A = Model(inputs=x, outputs=pred)

    model_A.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3, decay=0.001),
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"])
    return model_A


def mmoe_cnn2layer_model(n_classes, n1=128, n2=256, dropout_rate=0.2, input_shape=(28, 28), temperature=1.5,
                         num_experts=3):
    model_A, x = None, None
    x = Input(input_shape)
    if len(input_shape) == 2:
        input = Reshape((input_shape[0], input_shape[1], 1))(x)
    else:
        input = Reshape(input_shape)(x)

    experts = []
    for i in range(num_experts):
        y = Conv2D(filters=n1, kernel_size=(3, 3), strides=1, padding="same",
                   activation=None, name="Conv1_expert" + str(i))(input)
        y = BatchNormalization(name="BN1_expert" + str(i))(y)
        y = Activation("relu")(y)
        y = Dropout(dropout_rate)(y)
        y = AveragePooling2D(pool_size=(2, 2), strides=1, padding="same")(y)

        y = Conv2D(filters=n2, kernel_size=(2, 2), strides=2, padding="valid",
                   activation=None, name="Conv2_expert" + str(i))(y)
        y = BatchNormalization(name="BN2_expert" + str(i))(y)
        y = Activation("relu")(y)
        y = Dropout(dropout_rate)(y)
        y = AveragePooling2D(pool_size=(2, 2), strides=2, padding="valid")(y)

        y = Flatten()(y)
        experts.append(y)

    # gate
    input_flatten = Flatten()(input)
    g = Dense(units=num_experts, activation='relu', name='gate')(input_flatten)

    # attention
    att = Dense(units=200, activation='relu', use_bias=False, name='Att_hidden')(
        tf.stack(experts, axis=1))
    att = Dropout(0.5)(att)
    att = Dense(units=100, activation='relu', use_bias=False, name='Att_hidden2')(att)
    att = Dropout(0.5)(att)
    att = Dense(units=50, activation='relu', use_bias=False, name='Att_hidden3')(att)
    att = Dropout(0.5)(att)
    att = Dense(units=1, activation=None, use_bias=False, name='Att')(att)
    att = Dropout(0.5)(att)
    att = att
    attention_probs = Softmax(name='Att_softmax')(tf.squeeze(att, axis=-1))
    g = multiply([attention_probs, g], name='weight')
    g = Softmax(name='weight_softmax')(g)
    g = RepeatVector(experts[0].shape[-1])(g)

    Added_Embedding = []
    for i in range(num_experts):
        gi = multiply([g[:, :, i], experts[i]])
        Added_Embedding.append(gi)

    embedding = add(Added_Embedding, name='embeddings')

    # tower
    t = Dense(units=n_classes, activation=None, use_bias=False,
              kernel_regularizer=tf.keras.regularizers.l2(1e-3), name='logit')(embedding)  # embedding
    t = Activation("softmax", name='pred')(t)

    model_A = Model(inputs=x, outputs=t)

    model_A.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"])
    return model_A


def cnn_3layer_fc_model(n_classes, n1=128, n2=192, n3=256, dropout_rate=0.2, input_shape=(28, 28)):
    model_A, x = None, None

    x = Input(input_shape)
    if len(input_shape) == 2:
        y = Reshape((input_shape[0], input_shape[1], 1))(x)
    else:
        y = Reshape(input_shape)(x)
    y = Conv2D(filters=n1, kernel_size=(3, 3), strides=1, padding="same",
               activation=None, name='Conv1')(y)
    y = BatchNormalization(name='BN1')(y)
    y = Activation("relu")(y)
    y = Dropout(dropout_rate)(y)
    y = AveragePooling2D(pool_size=(2, 2), strides=1, padding="same")(y)

    y = Conv2D(filters=n2, kernel_size=(2, 2), strides=2, padding="valid",
               activation=None, name='Conv2')(y)
    y = BatchNormalization(name='BN2')(y)
    y = Activation("relu")(y)
    y = Dropout(dropout_rate)(y)
    y = AveragePooling2D(pool_size=(2, 2), strides=2, padding="valid")(y)

    y = Conv2D(filters=n3, kernel_size=(3, 3), strides=2, padding="valid",
               activation=None, name='Conv3')(y)
    y = BatchNormalization(name='BN3')(y)
    y = Activation("relu")(y)
    y = Dropout(dropout_rate)(y)

    y = Flatten(name='embeddings')(y)
    y = Dense(units=n_classes, activation=None, use_bias=False,
              kernel_regularizer=tf.keras.regularizers.l2(1e-3), name='fc1')(y)
    y = Activation("softmax", name='pred')(y)

    model_A = Model(inputs=x, outputs=y)

    model_A.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"])
    return model_A


def cnn_2layer_fc_model(n_classes, n1=128, n2=256, dropout_rate=0.2, input_shape=(28, 28)):
    model_A, x = None, None

    x = Input(input_shape)
    if len(input_shape) == 2:
        y = Reshape((input_shape[0], input_shape[1], 1))(x)
    else:
        y = Reshape(input_shape)(x)
    y = Conv2D(filters=n1, kernel_size=(3, 3), strides=1, padding="same",
               activation=None, name='Conv1')(y)
    y = BatchNormalization(name='BN1')(y)
    y = Activation("relu")(y)
    y = Dropout(dropout_rate)(y)
    y = AveragePooling2D(pool_size=(2, 2), strides=1, padding="same")(y)

    y = Conv2D(filters=n2, kernel_size=(3, 3), strides=2, padding="valid",
               activation=None, name='Conv2')(y)
    y = BatchNormalization(name='BN2')(y)
    y = Activation("relu")(y)
    y = Dropout(dropout_rate)(y)

    y = Flatten(name='embeddings')(y)
    y = Dense(units=n_classes, activation=None, use_bias=False,
              kernel_regularizer=tf.keras.regularizers.l2(1e-3), name='fc1')(y)
    y = Activation("softmax", name='pred')(y)

    model_A = Model(inputs=x, outputs=y)

    model_A.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"])
    return model_A


def remove_last_layer(model, loss="mean_absolute_error"):
    """
    Input: Keras model, a classification model whose last layer is a softmax activation
    Output: Keras model, the same model with the last softmax activation layer removed,
        while keeping the same parameters 
    """

    new_model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    new_model.set_weights(model.get_weights())
    new_model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3, decay=0.001),
                      loss=loss)

    return new_model


def train_models(models, X_train, y_train, X_test, y_test,
                 save_dir="./", save_names=None,
                 early_stopping=True, min_delta=0.001, patience=3,
                 batch_size=128, epochs=20, is_shuffle=True, verbose=1
                 ):
    '''
    Train an array of models on the same dataset. 
    We use early termination to speed up training. 
    '''

    resulting_val_acc = []
    record_result = []
    for n, model in enumerate(models):
        print("Training model ", n)
        if early_stopping:
            model.fit(X_train, y_train,
                      validation_data=(X_test, y_test),
                      # callbacks=[EarlyStopping(monitor='val_accuracy', min_delta=min_delta, patience=patience)],
                      batch_size=batch_size, epochs=epochs, shuffle=is_shuffle, verbose=verbose
                      )
        else:
            model.fit(X_train, y_train,
                      validation_data=(X_test, y_test),
                      batch_size=batch_size, epochs=epochs, shuffle=is_shuffle, verbose=verbose
                      )

        resulting_val_acc.append(model.history.history["val_accuracy"][-1])
        record_result.append({"train_acc": model.history.history["accuracy"],
                              "val_acc": model.history.history["val_accuracy"],
                              "train_loss": model.history.history["loss"],
                              "val_loss": model.history.history["val_loss"]})

        if save_dir is not None:
            save_dir_path = os.path.abspath(save_dir)
            # make dir
            try:
                os.makedirs(save_dir_path)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

            if save_names is None:
                file_name = save_dir + "model_{0}".format(n) + ".h5"
            else:
                file_name = save_dir + save_names[n] + ".h5"
            model.save(file_name)

    print("pre-train accuracy: ")
    print(resulting_val_acc)

    return record_result
