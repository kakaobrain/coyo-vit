import logging
import numpy as np
import scipy.ndimage
import tensorflow as tf


def set_mixed_precision_policy(strategy: tf.distribute.Strategy, use_mixed_precision: bool = True):
    if use_mixed_precision:
        if isinstance(strategy, tf.distribute.TPUStrategy):
            tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
        else:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
    else:
        tf.keras.mixed_precision.set_global_policy('float32')


def set_pretrained_pos_embed_for_vit(backbone, ckpt_path):
    reader = tf.train.load_checkpoint(ckpt_path)
    var_shape_map = reader.get_variable_to_shape_map()
    key = [key for key in var_shape_map if key.startswith('backbone/pos_emb') and not 'optimizer' in key]
    assert len(key) == 1, "cannot find positional embedding layer ('pos_emb')"
    posemb = reader.get_tensor(key[0])
    posemb_new = backbone.pos_emb.numpy()
    logging.info(f"load pretrained: resized variant: {posemb.shape} to {posemb_new.shape}")

    if posemb.shape[1] != posemb_new.shape[1]:
        ntok_new = posemb_new.shape[1] - 1
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]

        gs_old = int(np.sqrt(len(posemb_grid)))
        gs_new = int(np.sqrt(ntok_new))
        logging.info(f"load pretrained: grid-size from {gs_old} to {gs_new}")
        posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

        zoom = (gs_new / gs_old, gs_new / gs_old, 1)
        posemb_grid = scipy.ndimage.zoom(posemb_grid, zoom, order=1)
        posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
        embedding_weights = tf.convert_to_tensor(
            np.concatenate([posemb_tok, posemb_grid], axis=1)
        )
    else:
        embedding_weights = posemb
    backbone.pos_emb.assign(embedding_weights)
