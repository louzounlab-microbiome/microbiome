import tensorflow as tf
from tensorflow.python.keras import backend as K

tf.enable_eager_execution()
from tensorflow.python.keras import optimizers, regularizers, callbacks, losses
from Preprocess import tf_analaysis
PADDED_VALUE = -999



def my_loss(y_true, y_pred):
    mse_loss = my_mse_loss(y_true, y_pred)

    time_sense_loss = y_true[:, 2] - y_pred[:, 1]  # Max_delta - predicted_delta should be negative
    tsls = tf.math.maximum(0, time_sense_loss) #tf.square(time_sense_loss)

    return y_true[:, 4] *tsls + y_true[:, 3] * mse_loss


def my_loss_batch(y_true, y_pred):
    batch_size = y_pred.shape[0]
    steps_size = y_pred.shape[1]


    loss = 0
    total_samples_in_batches = 0
    for sample_in_batch in range(batch_size):
        single_y_true = y_true[sample_in_batch, :, :]
        single_y_pred = y_pred[sample_in_batch, :, :]

        mask = tf.reduce_all(tf.logical_not(tf.equal(single_y_true, PADDED_VALUE)),axis=1)

        single_y_true = tf.boolean_mask(single_y_true, mask)
        single_y_pred = tf.boolean_mask(single_y_pred, mask)

        loss_per_seq = my_loss(single_y_true, single_y_pred)
        loss += tf.reduce_sum(loss_per_seq)
        total_samples_in_batches += tf.convert_to_tensor(single_y_true[:, 1].shape[0], preferred_dtype=tf.float32)

    return loss / tf.cast(total_samples_in_batches, tf.float32)

def my_mse_loss(y_true, y_pred):
    mse_loss = tf.reduce_mean(losses.mean_squared_error(tf.expand_dims(y_true[:, 1], axis=-1), tf.expand_dims(y_pred[:, 1], axis=-1)))

    return mse_loss



def build_lstm_model(number_neurons_per_layer, l2_lambda, input_size, number_layers, dropout):
    test_model = tf_analaysis.nn_model()
    regularizer = regularizers.l2(l2_lambda)

    model_structure = [({'units': 10, 'input_shape': (None, input_size),
                         'return_sequences': True}, 'LSTM')]

    for layer_idx in range(number_layers):
        model_structure.append({'units': number_neurons_per_layer, 'activation': tf.nn.relu,
                                'kernel_regularizer': regularizer})
        model_structure.append(({'rate': dropout}, 'dropout'))

    model_structure.append({'units': 4, 'kernel_regularizer': regularizer})
    test_model.build_nn_model(hidden_layer_structure=model_structure)
    return test_model


def build_fnn_model(number_neurons_per_layer, l2_lambda, input_size, number_layers, dropout):
    # K.get_session().close()
    # K.set_session(tf.Session())

    test_model = tf_analaysis.nn_model()
    regularizer = regularizers.l2(l2_lambda)
    # model_structure = [({'units': input_size, 'input_shape': (input_size), 'activation': tf.nn.relu, 'kernel_regularizer': regularizer}, 'dense')]
    model_structure = [{'units': input_size, 'activation': tf.nn.relu, 'kernel_regularizer': regularizer, 'input_dim': input_size}]
    for layer_idx in range(number_layers):
        model_structure.append(
            {'units': number_neurons_per_layer, 'activation': tf.nn.relu, 'kernel_regularizer': regularizer})
        model_structure.append(({'rate': dropout}, 'dropout'))

    model_structure.append({'units': 4, 'kernel_regularizer': regularizer})
    test_model.build_nn_model(hidden_layer_structure=model_structure)
    # K.get_session().run(tf.global_variables_initializer())
    # test_model.model.fit()
    return test_model

def compile_model(model, loss, metrics):
    return model.compile_nn_model(loss=loss, metrics=metrics)


