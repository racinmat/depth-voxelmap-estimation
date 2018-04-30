import tensorflow as tf


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
    # print(tf.nn.softmax_cross_entropy_with_logits(
    #             labels=labels_obstacles,
    #             logits=logits).shape)
    # loss = tf.reduce_mean(
    #         tf.nn.weighted_cross_entropy_with_logits(
    #             targets=labels_obstacles,
    #             logits=logits,
    #             pos_weight=tf.not_equal(labels, - tf.ones_like(labels))),
    #     )
    # loss = tf.reduce_mean(
    #     tf.multiply(
    #         tf.nn.softmax_cross_entropy_with_logits(
    #             labels=labels_obstacles,
    #             logits=logits),
    #         tf.not_equal(labels, - tf.ones_like(labels))
    #     )
    # )
    loss = tf.reduce_mean(
        tf.losses.mean_squared_error(
            labels=labels_obstacles,
            predictions=tf.nn.softmax(logits),
            weights=tf.not_equal(labels, - tf.ones_like(labels))
        )
    )
    # masking by weighted crossentropy, unknown values give us 0, known values give us 1
    return tf.identity(loss, 'loss')
