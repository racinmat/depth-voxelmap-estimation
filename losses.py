import tensorflow as tf

OBSTACLE_THRESHOLD = 0.6    # probably need to tune it later, used when output is not softmaxed


def is_obstacle(voxel):
    return voxel >= OBSTACLE_THRESHOLD


def is_free(voxel):
    return (voxel < OBSTACLE_THRESHOLD) & (voxel >= 0)  # just check o be compatible with unknown voxels


def tf_labels_to_info_gain(labels, logits, alpha=0.2):
    # int 16 stačí, protože je to index binu pro hloubku
    last_axis = len(logits.shape) - 1
    label_idx = tf.expand_dims(tf.argmax(labels, axis=last_axis), last_axis)
    label_idx = tf.cast(label_idx, dtype=tf.int32)
    # expanding back to have size in dim 4 (reduced by argmax)
    tiling_shape = list(labels.shape)
    tiling_shape[0:last_axis] = [1 for i in range(last_axis)]
    tiling_shape[last_axis] = tiling_shape[last_axis].value
    label_idx = tf.tile(label_idx, tiling_shape)
    prob_bin_idx = tf.range(logits.shape[last_axis], dtype=tf.int32)
    for i in range(last_axis):
        prob_bin_idx = tf.expand_dims(prob_bin_idx, 0)
    # prob_bin_idx = tf.transpose(prob_bin_idx)
    tiling_shape = [i.value for i in labels.shape]
    tiling_shape[0] = tf.shape(labels)[0]
    tiling_shape[last_axis] = 1
    prob_bin_idx = tf.tile(prob_bin_idx, tiling_shape)

    difference = (label_idx - prob_bin_idx) ** 2
    difference = tf.cast(difference, dtype=tf.float32)
    info_gain = tf.exp(-alpha * difference)
    return info_gain


def information_gain_loss(labels, logits, alpha=0.2):
    # unknown voxels have -1 values, so we unify it with free voxels here for BC
    labels_obstacles = tf.maximum(labels, tf.zeros_like(labels))
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=tf_labels_to_info_gain(labels=labels_obstacles, logits=logits, alpha=alpha),
            logits=logits))
    return tf.identity(loss, 'loss')


def information_gain_loss_with_undefined(labels, logits, alpha=0.2):
    # todo: fix, wrong dimensions, does not work
    # unknown voxels have -1 values, so we unify it with free voxels here for BC
    labels_obstacles = tf.maximum(labels, tf.zeros_like(labels))
    loss = tf.reduce_mean(
        tf.nn.weighted_cross_entropy_with_logits(
            targets=tf_labels_to_info_gain(labels=labels_obstacles, logits=logits, alpha=alpha),
            logits=logits,
            pos_weight=tf.not_equal(labels, - tf.ones_like(labels))
        )
    )
    return tf.identity(loss, 'loss')


def softmax_loss(labels, logits):
    labels_obstacles = tf.maximum(labels, tf.zeros_like(labels))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels_obstacles, logits=logits))
    return tf.identity(loss, 'loss')


def l2_voxelwise_loss_with_undefined(labels, logits):
    # unknown voxels have -1 values, so we unify it with free voxels here for BC
    # print(labels.shape)
    print(logits.shape)
    labels_obstacles = tf.maximum(labels, tf.zeros_like(labels))
    print(labels_obstacles.shape)
    print(tf.not_equal(labels, - tf.ones_like(labels)).shape)
    loss = tf.reduce_mean(
        tf.losses.mean_squared_error(
            labels=labels_obstacles,
            predictions=tf.nn.softmax(logits),
            weights=tf.not_equal(labels, - tf.ones_like(labels))
        )
    )
    # masking by weighted crossentropy, unknown values give us 0, known values give us 1
    return tf.identity(loss, 'loss')


def get_known_mask(labels):
    return tf.cast(tf.not_equal(labels, - tf.ones_like(labels)), dtype=tf.float32)


def logistic_voxelwise_loss_with_undefined(labels, predicted):
    # here I don't use thresholding
    # this loss is class balanced
    # unknown voxels have -1 values, so we unify it with free voxels here for BC
    # print(labels.shape)
    print(predicted.shape)
    labels_shifted = tf.where(tf.equal(labels, tf.zeros_like(labels)), - tf.ones_like(labels), labels)  # so 1 is obstacle and -1 is free
    known_mask = get_known_mask(labels)
    # now I weight classes so all weights sum to one and after weighting they are balanced
    # 0.5 weight comes to free voxels
    # 0.5 weight comes to occupied voxels
    occupied_voxels_num = tf.reduce_sum(tf.equal(labels, 1), [1, 2, 3], keep_dims=True)
    free_voxels_num = tf.reduce_sum(tf.equal(labels, 0), [1, 2, 3], keep_dims=True)
    occupied_mask = 1 / (2 * occupied_voxels_num)
    free_mask = 1 / (2 * free_voxels_num)
    known_mask = tf.where(tf.equal(labels, 1), occupied_mask, known_mask)
    known_mask = tf.where(tf.equal(labels, 0), free_mask, known_mask)
    print(labels_shifted.shape)
    print(labels_shifted.dtype)
    print(known_mask.dtype)
    loss = tf.reduce_mean(known_mask * tf.log(tf.ones_like(labels) + tf.exp(- labels_shifted * predicted)))
    return tf.identity(loss, 'loss')


def softmax_voxelwise_loss_with_undefined(labels, predicted):
    # loss from https://arxiv.org/pdf/1604.00449.pdf, but I use reduce
    # unknown voxels have -1 values, so we unify it with free voxels here for BC
    # because I use equality to obstacle and free, I don't need masking
    # to be independent on batch size, I sum all voxels per sample, but mean per samples in batch
    print(predicted.shape)
    loss = tf.reduce_mean(tf.reduce_sum(tf.equal(labels, 1) * tf.log(predicted) + tf.equal(labels, 0) * tf.log(1 - predicted), [1, 2, 3]))
    return tf.identity(loss, 'loss')
