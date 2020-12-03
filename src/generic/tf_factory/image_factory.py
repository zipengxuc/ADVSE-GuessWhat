import tensorflow as tf

from neural_toolbox.cbn_pluggin import CBNfromLSTM
from neural_toolbox.resnet import create_resnet
from neural_toolbox.cbn import ConditionalBatchNorm


def get_image_features(image, is_training, config, resnet_scope="", cbn=None):
    image_input_type = config["image_input"]

    # Extract feature from 1D-image feature s
    if image_input_type == "fc8" \
            or image_input_type == "fc7" \
            or image_input_type == "dummy":

        image_out = image
        if config.get('normalize', False):
            image_out = tf.nn.l2_normalize(image, dim=1, name="fc_normalization")

    elif image_input_type == "rcnn":

        img = tf.transpose(image, perm=[0, 2, 1])  # B, 2048, 36
        att = tf.nn.softmax(tf.ones([tf.shape(img)[0], 1, 36]), axis=-1)
        image_out = tf.reduce_sum(img * att, axis=-1)  # B, 2048
        if config.get('normalize', False):
            image_out = tf.nn.l2_normalize(image_out, dim=1, name="fc_normalization")

    elif image_input_type.startswith("conv") or image_input_type.startswith("raw"):

        # Extract feature from raw images
        if image_input_type.startswith("raw"):

            # Create ResNet
            resnet_version = config['resnet_version']
            image_out = create_resnet(image,
                                      is_training=is_training,
                                      scope=resnet_scope,
                                      cbn=cbn,
                                      resnet_version=resnet_version,
                                      resnet_out=config.get('resnet_out', "block4"))
        else:
            image_out = image

        if config.get('normalize', False):
            image_out = tf.nn.l2_normalize(image_out, dim=[1, 2, 3])

    else:
        assert False, "Wrong input type for image"

    return image_out


def get_cbn(config, question, dropout_keep, is_training):

    cbn_factory = CBNfromLSTM(question, no_units=config['cbn']["cbn_embedding_size"], dropout_keep=dropout_keep)

    excluded_scopes = config["cbn"].get('excluded_scope_names', [])
    cbn = ConditionalBatchNorm(cbn_factory, excluded_scope_names=excluded_scopes, is_training=is_training)

    return cbn
