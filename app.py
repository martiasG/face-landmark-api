from flask import Flask, send_file
app = Flask(__name__)
from flask import Flask, jsonify, request
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import Concatenate, AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
import keras.backend as K
from keras.activations import relu
K.set_image_data_format('channels_last')


app = Flask(__name__)


def image(i, image_array, coordenates):
    img = image_array[i]
    plt.imshow(img.reshape(96, 96))
    pt = np.vstack(np.split(coordenates[i], 15)).T
    plt.scatter(pt[0], pt[1], c='red', marker='*')
    plt.savefig('image_result.jpg')


def inception_block(x, name, channels):
    """
    Implementation of an inception block
    """

    x_1x1 = Conv2D(channels['1x1'], (1, 1), padding='same', data_format='channels_last',
                   name='inception_3' + name + '_1x1_conv1')(x)
    x_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3' + name + '_1x1_bn1')(x_1x1)
    x_1x1 = Activation(activation=lambda x: relu(x, max_value=96))(x_1x1)

    x_3x3_reduce = Conv2D(channels['3x3_reduce'], (1, 1), padding='same', data_format='channels_last',
                          name='inception_3' + name + '_3x3_reduce')(x)
    x_3x3_reduce = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3' + name + '_3x3_reduce_bn2')(
        x_3x3_reduce)
    x_3x3_reduce = Activation(activation=lambda x: relu(x, max_value=96))(x_3x3_reduce)
    x_3x3 = Conv2D(channels['3x3'], (1, 1), padding='same', data_format='channels_last',
                   name='inception_3' + name + '_3x3_conv2')(x_3x3_reduce)
    x_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3' + name + '_3x3_bn2')(x_3x3)
    x_3x3 = Activation(activation=lambda x: relu(x, max_value=96))(x_3x3)

    x_5x5_reduce = Conv2D(channels['5x5_reduce'], (1, 1), data_format='channels_last',
                          name='inception_3' + name + '_5x5_reduce')(x)
    x_5x5_reduce = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3' + name + '_5x5_reduce_bn1')(
        x_5x5_reduce)
    x_5x5_reduce = Activation(activation=lambda x: relu(x, max_value=96))(x_5x5_reduce)
    x_5x5 = ZeroPadding2D(padding=(2, 2), data_format='channels_last')(x_5x5_reduce)
    x_5x5 = Conv2D(channels['5x5'], (5, 5), data_format='channels_last', name='inception_3' + name + '_5x5_conv3')(
        x_5x5)
    x_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3' + name + '_5x5_bn3')(x_5x5)
    x_5x5 = Activation(activation=lambda x: relu(x, max_value=96))(x_5x5)

    x_pool = MaxPooling2D(pool_size=3, strides=1, data_format='channels_last')(x)
    x_pool_proj = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(x_pool)
    x_pool_proj = Conv2D(channels['pool_proj'], (1, 1), data_format='channels_last',
                         name='inception_3' + name + '_pool_conv')(x_pool_proj)
    x_pool_proj = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3' + name + '_pool_bn')(x_pool_proj)
    x_pool_proj = Activation(activation=lambda x: relu(x, max_value=96))(x_pool_proj)

    # CONCAT
    inception = Concatenate()([x_3x3, x_5x5, x_pool_proj, x_1x1])
    return inception


def model_land_mark_detection_inception(input_shape):
    x_train_input = Input(input_shape)

    x = Conv2D(64, (7, 7), strides=(2, 2), data_format='channels_last')(x_train_input)
    x = Activation(activation=lambda x: relu(x, max_value=96))(x)

    x = MaxPooling2D(pool_size=3, strides=1, data_format='channels_last')(x)

    x = Conv2D(192, (3, 3), strides=(1, 1), data_format='channels_last')(x)
    x = Activation(activation=lambda x: relu(x, max_value=96))(x)

    x = MaxPooling2D(pool_size=3, strides=1, data_format='channels_last')(x)

    dict_c = {'1x1': 64, '3x3_reduce': 96, '3x3': 128, '5x5_reduce': 16, '5x5': 32, 'pool_proj': 32}
    x = inception_block(x, name='a', channels=dict_c)
    dict_c = {'1x1': 128, '3x3_reduce': 128, '3x3': 192, '5x5_reduce': 32, '5x5': 96, 'pool_proj': 64}
    x = inception_block(x, name='b', channels=dict_c)
    x = MaxPooling2D(pool_size=3, strides=2, data_format='channels_last')(x)

    dict_c = {'1x1': 192, '3x3_reduce': 96, '3x3': 208, '5x5_reduce': 16, '5x5': 48, 'pool_proj': 64}
    x = inception_block(x, name='c', channels=dict_c)
    dict_c = {'1x1': 160, '3x3_reduce': 112, '3x3': 224, '5x5_reduce': 24, '5x5': 64, 'pool_proj': 64}
    x = inception_block(x, name='d', channels=dict_c)

    dict_c = {'1x1': 128, '3x3_reduce': 128, '3x3': 256, '5x5_reduce': 24, '5x5': 64, 'pool_proj': 64}
    x = inception_block(x, name='e', channels=dict_c)
    dict_c = {'1x1': 112, '3x3_reduce': 144, '3x3': 288, '5x5_reduce': 32, '5x5': 64, 'pool_proj': 64}
    x = inception_block(x, name='f', channels=dict_c)
    dict_c = {'1x1': 256, '3x3_reduce': 160, '3x3': 320, '5x5_reduce': 32, '5x5': 128, 'pool_proj': 128}
    x = inception_block(x, name='g', channels=dict_c)

    x = MaxPooling2D(pool_size=3, strides=2, data_format='channels_last')(x)

    dict_c = {'1x1': 256, '3x3_reduce': 160, '3x3': 320, '5x5_reduce': 32, '5x5': 128, 'pool_proj': 128}
    x = inception_block(x, name='h', channels=dict_c)
    dict_c = {'1x1': 384, '3x3_reduce': 192, '3x3': 384, '5x5_reduce': 48, '5x5': 128, 'pool_proj': 128}
    x = inception_block(x, name='r', channels=dict_c)

    x = AveragePooling2D(pool_size=7, strides=1, data_format='channels_last')(x)

    x = Flatten()(x)
    x = Dense(1024, activation=lambda x: relu(x, max_value=96))(x)
    x = Dropout(0.4)(x)

    x = Dense(30, activation=lambda x: relu(x, max_value=96))(x)

    model = Model(inputs=x_train_input, outputs=x)

    return model


def load_model():
    """Load and return the model"""
    # TODO: INSERT CODE
    model = model_land_mark_detection_inception((96, 96, 1))
    model.load_weights('weights_inception_fillna_100_epochs.h5')
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    return model


# you can then reference this model object in evaluate function/handler
model = load_model()


# The request method is POST (this method enables your to send arbitrary data to the endpoint in the request body, including images, JSON, encoded-data, etc.)
@app.route('/', methods=["POST"])
def evaluate():
    from scipy import ndimage

    # TODO: data/input preprocessing
    file = request.files.get('file')
    # eg: request.args.get('style')
    # eg: request.form.get('model_name')

    predict_images = []
    pic = np.array(ndimage.imread(file, flatten=False))
    predict_images.append(pic.reshape(96, 96, 1))
    # TODO: model evaluation

    predictions = model.predict(np.array(predict_images))

    image(0, np.array(predict_images), predictions)

    # TODO: return prediction
    return send_file('image_result.jpg')


# The following is for running command `python app.py` in local development, not required for serving on FloydHub.
if __name__ == "__main__":
    print("* Starting web server... please wait until server has fully started")
    app.run(host='0.0.0.0', threaded=False)
