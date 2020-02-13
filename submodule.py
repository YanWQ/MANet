"""
# ==================================
# AUTHOR : Yan Li, Qiong Wang
# CREATE DATE : 02.10.2020
# Contact : liyanxian19@gmail.com
# ==================================
# Change History: None
# ==================================
"""
############################################################
#  Import third-party libs (numpy, tensorflow)
############################################################
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Layer, Input, Lambda

from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Conv3D, Conv3DTranspose
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import Dropout, BatchNormalization
from tensorflow.python.keras.layers import concatenate, add
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import backend as K



k_init = "glorot_uniform"
norm_type = "bn"


############################################################
#  Define ops and layers
############################################################
def conv_2d(x, filter_num, ks, strides=None, padding=None, dilation_rate=None, use_bias=None,
            conv_type=None, activation=None, k_reg=None, name=None):
    if not isinstance(ks, (list, tuple)):
        ks = (ks, ks)

    if strides is None:
        strides = (1, 1)

    if padding == "zero":
        padding = "same"
    else:
        padding = "valid"

    if dilation_rate is None:
        dilation_rate = 1

    if use_bias is None:
        use_bias = True

    if k_reg is not None:
        weight_decay = 0.005
        k_reg = regularizers.l2(weight_decay)

    if conv_type == "conv" or conv_type is None:
        x = Conv2D(filter_num, ks, strides=strides, padding=padding,
                   dilation_rate=dilation_rate, use_bias=use_bias, kernel_initializer=k_init,
                   kernel_regularizer=k_reg, name=name)(x)

    if activation is not None:
        if activation == "relu":
            x = Activation('relu')(x)
    return x

def conv_3d(x, filter_num, ks=None, strides=None, padding=None, use_bias=None,
            activation=None, k_reg=None, name=None):
    if ks is None:
        ks = 1

    if strides is None:
        strides = (1, 1, 1)

    if padding == "zero":
        padding = "same"

    if use_bias is None:
        use_bias = True

    if k_reg is not None:
        weight_decay = 0.005
        k_reg = regularizers.l2(weight_decay)

    x = Conv3D(filter_num, ks, strides=strides, padding=padding, use_bias = use_bias,
               kernel_initializer=k_init, kernel_regularizer=k_reg, name=name)(x)

    if activation is not None:
        if activation == "relu":
            x = Activation('relu')(x)
    return x

def dconv_3d(x, filter_num, ks, strides=None, padding=None, activation=None, name=None):
    if strides is None:
        strides = (1, 1, 1)

    x = Conv3DTranspose(filter_num, ks, strides=strides, padding=padding,
                        kernel_initializer=k_init, name=name)(x)

    if activation is not None:
        if activation == "relu":
            x = Activation('relu')(x)
    return x

def act(x, activation=None):
    if activation == "relu":
        x = Activation('relu')(x)
    return x

def normal(x, normal_type=None):
    if normal_type is not None:
        if normal_type == "bn":
            x = BatchNormalization(axis=-1)(x)
    return x



def bilinear_sampler(x, v):

    def get_grid_array(N, H, W, h, w):
        N_i = tf.range(N)
        H_i = tf.range(h+1, h+H+1)
        W_i = tf.range(w+1, w+W+1)
        n, h, w, = tf.meshgrid(N_i, H_i, W_i, indexing='ij')
        n = tf.expand_dims(n, axis=3)
        h = tf.expand_dims(h, axis=3)
        w = tf.expand_dims(w, axis=3)

        n = tf.cast(n, tf.float32)
        h = tf.cast(h, tf.float32)
        w = tf.cast(w, tf.float32)

        return n, h, w

    shape = tf.shape(x)
    N = shape[0]
    H_ = H = shape[1]
    W_ = W = shape[2]
    h = w = 0

    vy, vx = tf.split(v, 2, axis=3)

    n, h, w = get_grid_array(N, H, W, h, w) # [N, H, W, 3]

    vx0 = tf.floor(vx)
    vy0 = tf.floor(vy)
    vx1 = vx0 + 1
    vy1 = vy0 + 1 # [N, H, W, 1]


    H_1 = tf.cast(H_-1, tf.float32)
    W_1 = tf.cast(W_-1, tf.float32)
    iy0 = tf.clip_by_value(vy0, 0., H_1)
    iy1 = tf.clip_by_value(vy1, 0., H_1)
    ix0 = tf.clip_by_value(vx0, 0., W_1)
    ix1 = tf.clip_by_value(vx1, 0., W_1)

    i00 = tf.concat([n, iy0, ix0], 3)
    i01 = tf.concat([n, iy1, ix0], 3)
    i10 = tf.concat([n, iy0, ix1], 3)
    i11 = tf.concat([n, iy1, ix1], 3) # [N, H, W, 3]
    i00 = tf.cast(i00, tf.int32)
    i01 = tf.cast(i01, tf.int32)
    i10 = tf.cast(i10, tf.int32)
    i11 = tf.cast(i11, tf.int32)

    x00 = tf.gather_nd(x, i00)
    x01 = tf.gather_nd(x, i01)
    x10 = tf.gather_nd(x, i10)
    x11 = tf.gather_nd(x, i11)
    w00 = tf.cast((vx1 - vx) * (vy1 - vy), tf.float32)
    w01 = tf.cast((vx1 - vx) * (vy - vy0), tf.float32)
    w10 = tf.cast((vx - vx0) * (vy1 - vy), tf.float32)
    w11 = tf.cast((vx - vx0) * (vy - vy0), tf.float32)
    output = tf.add_n([w00*x00, w01*x01, w10*x10, w11*x11])

    return output

def compute_cost_volume(inputs, t_s_ids, min_disp=None, max_disp=None, labels=None,
                        move_path=None, logger=None):
    # 1.initialization
    if isinstance(inputs, (list,)):
        View_n = len(inputs)
    _, H, W, C = K.int_shape(inputs[int((View_n - 1) / 2)]) # batch, height, width, channel of features

    # reference (-> central view)
    reference = inputs[int((View_n - 1) / 2)]
    # init spatial coordinates (of central view)
    cords = tf.zeros_like(inputs[0])[..., :2]
    # angular coordinate (of central view)
    cent_t_id = int((View_n - 1) / 2)
    cent_s_id = int((View_n - 1) / 2)

    # 1 label = the number of disparity
    label_disp = (max_disp - min_disp) / labels

    # camera moving/image view path
    if move_path == "LT":
        t_sign = 1
        s_sign = 1

    # cost list (contains a list of cost slices at all disparity offsets)
    cost_l = []

    # 2.iterate from the minimum to maximum disparity by every disparity unit (label_disp)
    for disp in np.arange(min_disp, max_disp, label_disp):
        feature_maps = []
        id = 0
        for t_id, s_id in zip(t_s_ids[0], t_s_ids[1]):
            # append (translated) feature maps
            if t_id == cent_t_id and s_id == cent_s_id:
                # self
                feature_maps.append(reference)
                continue
            else:
                # add 1 dimension: W; tile: repeat to W columns.
                tmp0 = tf.cast(tf.tile(tf.expand_dims(tf.clip_by_value(tf.range(H) + t_sign*(cent_t_id-t_id)*disp, 0, H), 1),
                                       [1, W]),
                               tf.float32)
                # add 0 dimension: H
                tmp1 = tf.cast(tf.tile(tf.expand_dims(tf.clip_by_value(tf.range(W) + s_sign*(cent_s_id-s_id)*disp, 0, W), 0),
                                       [H, 1]),
                               tf.float32)
                cords = tf.cast(tf.tile(tf.expand_dims(tf.stack([tmp0, tmp1], axis=2), axis=0),
                                        [tf.shape(cords)[0], 1, 1, 1]),
                               tf.float32)
                # shift: shift/translate feature maps by disparities
                target = Lambda(lambda x: bilinear_sampler(*x))([inputs[id], cords])

                # others
                feature_maps.append(target)

            id += 1

        # get a cost slice
        cost = concatenate(feature_maps, name='cost_d')# DxHxWx(NC)
        cost_l.append(cost)

    # 3.stack all cost slices to get a [cost volume]
    cost_volume = K.stack(cost_l, axis=1)

    return cost_volume

def cost_aggregation(x, ca_paras=None):

    base_num_filters = ca_paras["filter"]
    ksize = ca_paras["ks"]
    ds_stride = ca_paras["stride"]
    padding = ca_paras["padding"]
    num_down_conv = ca_paras["n_dc"]
    activ = ca_paras["activation"]
    down_convs = list()

    cna_paras = {'ks': ksize, 'stride': 1, 'padding': padding, 'filter': [base_num_filters]*2,
                 'activation': activ, 'layer_nums': 2}
    conv = cna_3b(x, cna_paras=cna_paras)
    down_convs.insert(0, conv)

    for i in range(num_down_conv):
        if i <= num_down_conv - 1:
            mult = 2
        else:
            mult = 4
        conv = cds_3b(conv, mult * base_num_filters, ksize, ds_stride, padding)
        down_convs.insert(0, conv)
    up_convs = down_convs[0]
    dcna_paras = {'ks': ksize, 'stride': ds_stride, 'padding': padding, 'filter': [base_num_filters],
                 'activation': activ, 'layer_nums': 1}
    for i in range(num_down_conv):
        dcna_paras["filter"][0] = K.int_shape(down_convs[i + 1])[-1]
        deconv = dcna_3b(up_convs, dcna_paras=dcna_paras)
        up_convs = add([deconv, down_convs[i + 1]])
    cost = dconv_3d(up_convs, 1, ksize, strides=ds_stride, padding=padding)
    aggre_cost = Lambda(lambda x: -x)(cost)

    return aggre_cost

def slicing(x, index, index_end=None, interval=1):
    if index_end is None:
        return x[..., index:index+1:interval]
    else:
        return x[..., index:index_end:interval]

def soft_min_reg(cv, axis=None, min_disp=None, max_disp=None, labels=None):
    if axis == 1:
        cv = Lambda(lambda x: K.squeeze(x, axis=-1))(cv)
    disp_map = K.reshape(K.arange(min_disp, max_disp - 0.000001, (max_disp - min_disp)/labels, dtype="float32"),
                         (1, 1, labels, 1))
    if axis == 1:
        output = K.conv2d(cv, disp_map, strides=(1, 1), padding='valid', data_format="channels_first")
        x = K.expand_dims(K.squeeze(output, axis=1), axis=-1)
    else:
        x = K.conv2d(cv, disp_map, strides=(1, 1), padding='valid')
    return x

class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        return (s[0], s[1] + self.padding[0][0] + self.padding[0][1],
                s[2] + self.padding[1][0] + self.padding[1][1], s[3])

    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad[0], h_pad[1]], [w_pad[0], w_pad[1]], [0, 0] ], 'REFLECT')



######################################################
#  Define blocks
######################################################
def cna_2b(x, cna_paras):
    ks = (cna_paras["ks"], cna_paras["ks"])
    strides = (cna_paras["stride"], cna_paras["stride"])
    padding = cna_paras["padding"]
    filt_nums = cna_paras["filter"]
    activation = cna_paras["activation"]
    for cnt in range(cna_paras["layer_nums"]):
        x = conv_2d(x, filt_nums[cnt], ks, strides=strides, padding=padding)
        x = normal(x, normal_type=norm_type)
        x = act(x, activation)

        if "dropout" in cna_paras.keys():
            x = Dropout(cna_paras["dropout"])(x)
    return x

def cna_3b(x, cna_paras):
    if isinstance(cna_paras["ks"], int):
        ks = (cna_paras["ks"], cna_paras["ks"], cna_paras["ks"])
    else:
        ks = cna_paras["ks"]

    strides = (cna_paras["stride"], cna_paras["stride"], cna_paras["stride"])
    padding = cna_paras["padding"]
    filt_nums = cna_paras["filter"]
    if "activation" in cna_paras.keys():
        activation = cna_paras["activation"]
    else:
        activation = "relu" # default
    for cnt in range(cna_paras["layer_nums"]):
        x = conv_3d(x, filt_nums[cnt], ks, strides=strides, padding=padding)
        x = normal(x, normal_type=norm_type)
        x = act(x, activation)
    return x

def cds_3b(x, filters, ksize, ds_stride, padding):
    cna_paras = {'ks': ksize, 'stride': ds_stride, 'padding': padding, 'filter': [filters],
                 'activation': "relu", 'layer_nums': 1}
    conv = cna_3b(x, cna_paras)
    cna_paras['stride'] = 1
    conv = cna_3b(conv, cna_paras)
    conv = cna_3b(conv, cna_paras)
    return conv

def dcna_3b(x, dcna_paras):
    ks = (dcna_paras["ks"], dcna_paras["ks"], dcna_paras["ks"])
    strides = (dcna_paras["stride"], dcna_paras["stride"], dcna_paras["stride"])
    padding = dcna_paras["padding"]
    filt_nums = dcna_paras["filter"]
    activation = dcna_paras["activation"]
    for cnt in range(dcna_paras["layer_nums"]):
        x = dconv_3d(x, filt_nums[cnt], ks, strides=strides, padding=padding)
        x = normal(x, normal_type=norm_type)
        x = act(x, activation)
    return x



######################################################
#  Define modules
######################################################
def cna_m(x, cna_configs, layer_names='random', conv_dims=2):
    if layer_names == "random":
        ks = (cna_configs["ks"], cna_configs["ks"])
        stride = cna_configs["stride"]
        padding = cna_configs["padding"]
        filt_nums = cna_configs["filter"]
        activation = cna_configs["activation"]
        conv_type = cna_configs["conv_type"]

        for cnt in range(cna_configs["layer_nums"]):
            if conv_dims == 2:
                x = conv_2d(x, filt_nums[cnt], ks, padding=padding, conv_type=conv_type)
            elif conv_dims == 3:
                x = conv_3d(x, filt_nums[cnt], ks, padding=padding)
            x = normal(x, normal_type=norm_type)
            x = act(x, activation)
    return x

def feature_extraction_m(input_shape, feat_paras=None):
    x = Input(shape=input_shape)
    ys = []

    # feature
    if 'pyr' in feat_paras.keys():
        pyr = feat_paras['pyr']
    else:
        pyr = False
    ks = (feat_paras['ks'], feat_paras['ks'])
    padding = feat_paras['padding']
    ret_feat_levels = feat_paras['ret_feat_levels']
    stride = feat_paras['stride']
    filt_nums = feat_paras['filter']
    for cnt in range(feat_paras['layer_nums']):
        cna_paras = {"filter": [feat_paras["filter"][cnt]]*1, "ks": feat_paras["ks"], "stride": stride[1], "padding": padding,
                     "activation": feat_paras["activation"], "layer_nums": 1}
        if cnt == 0:
            y = x
        # downsample
        y = conv_2d(y, filt_nums[cnt], ks, strides=(stride[0], stride[0]), padding=padding)
        # conv
        y = cna_2b(y, cna_paras)
        if pyr and cnt >= (feat_paras['layer_nums'] - ret_feat_levels):
            ys.append(y)
    if not pyr:
        ys.append(y)

    sfm = Model(inputs=[x], outputs=ys) # shared feature module

    return sfm
