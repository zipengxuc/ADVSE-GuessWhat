import tensorflow as tf

from neural_toolbox.attention import compute_attention, compute_glimpse, compute_convolution_pooling
from neural_toolbox.attention import compute_conv2d, compute_linear_att, compute_glimpse2, compute_glimpse3


def get_attention(feature_map, context, config, is_training, dropout_keep, reuse=False):
    attention_mode = config.get("mode", None)

    if attention_mode == "none":
        image_out = feature_map

    elif attention_mode == "conv2d":
        image_out = compute_conv2d(feature_map,
                                   context,
                                   no_mlp_units=config['no_attention_mlp'],
                                   keep_dropout=dropout_keep,
                                   reuse=reuse)

    elif attention_mode == "linear":
        image_out = compute_linear_att(feature_map,
                                       context,
                                       no_mlp_units=config['no_attention_mlp'],
                                       fuse_mode=config['fuse_mode'],
                                       keep_dropout=dropout_keep,
                                       reuse=reuse)

    elif attention_mode == "max":
        image_out = tf.reduce_max(feature_map, axis=(1, 2))

    elif attention_mode == "mean":
        image_out = tf.reduce_mean(feature_map, axis=(1, 2))

    elif attention_mode == "classic":
        image_out = compute_attention(feature_map,
                                      context,
                                      no_mlp_units=config['no_attention_mlp'],
                                      fuse_mode=config['fuse_mode'],
                                      keep_dropout=dropout_keep,
                                      reuse=reuse)

    elif attention_mode == "glimpse":
        image_out = compute_glimpse(feature_map,
                                    context,
                                    no_glimpse=config['no_glimpses'],
                                    glimpse_embedding_size=config['no_attention_mlp'],
                                    keep_dropout=dropout_keep,
                                    is_training=is_training,
                                    reuse=reuse)

    elif attention_mode == "glimpse2":
        image_out = compute_glimpse2(feature_map,
                                    context,
                                    no_glimpse=config['no_glimpses'],
                                    glimpse_embedding_size=config['no_attention_mlp'],
                                    keep_dropout=config['drop_out_keep'],
                                    is_training=is_training,
                                    reuse=reuse)
    elif attention_mode == "glimpse3":
        image_out = compute_glimpse3(feature_map,
                                    context,
                                    no_glimpse=config['no_glimpses'],
                                    glimpse_embedding_size=config['no_attention_mlp'],
                                    keep_dropout=config['drop_out_keep'],
                                    is_training=is_training,
                                    reuse=reuse)
    elif attention_mode == "conv_pooling":
        image_out = compute_convolution_pooling(feature_map,
                                                no_mlp_units=config['no_attention_mlp'],
                                                is_training=is_training,
                                                reuse=reuse)

    else:
        assert False, "Wrong attention mode: {}".format(attention_mode)

    return image_out
