"""
# ==================================
# AUTHOR : Yan Li, Qiong Wang
# CREATE DATE : 02.10.2020
# Contact : liyanxian19@gmail.com
# ==================================
# Change History: None
# ==================================
"""
########## Import python libs ##########
import math

########## Import third-party libs ##########
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Lambda
from tensorflow.python.keras.layers import UpSampling3D
from tensorflow.python.keras.layers import Average, concatenate

########## Import our libs ##########
from submodule import *



def manet(input_layer_names, input_shape, config=None, logger=None):
    output_layers = []

    # paras of multi-scales
    nb_pyr_levels = 3
    pyr_levels = list(range(nb_pyr_levels))
    ret_feat_levels = 2

    # paras of layers
    conv_type, ks, activ = "conv", 2, "relu"

    # paras of cost volume (cv)
    min_disp, max_disp, num_disp_labels = -4, 4, 80

    ########## input layers ##########
    input_layers = []
    for _, input_layer_name in enumerate(input_layer_names):
        x = Input(shape=input_shape, name=input_layer_name)
        input_layers.append(x)


    ########## Branch_2, 3: cv ##########
    pyr_outputs = [] # outputs of pyramid level 1, 2

    # 1. Feature extraction
    nb_filt1 = 8
    feature_s_paras = {'ks': ks, 'stride': [2, 1], 'padding': "zero", 'filter': [nb_filt1, nb_filt1*2]*1,
                       'activation': activ, 'conv_type': conv_type, 'pyr': True, 'layer_nums': 2, "ret_feat_levels": ret_feat_levels}
    feature_s_m = feature_extraction_m((input_shape[0], input_shape[1], 1), feat_paras=feature_s_paras)

    fs_ts_ids = []
    feature_streams = []
    for stream_id, x in enumerate(input_layers):
        if stream_id > 1:
            continue
        feature_stream = []
        for x_sid in range(input_shape[2]):
            x_sub = Lambda(slicing, arguments={'index': x_sid})(x)
            x_sub = feature_s_m(x_sub)
            feature_stream.append(x_sub)

        if stream_id == 0:
            t_ids = list(range(input_shape[2]))[::-1]
            s_ids = [int((input_shape[2]-1)/2)]*input_shape[2]
        elif stream_id == 1:
            t_ids = [int((input_shape[2]-1)/2)]*input_shape[2]
            s_ids = list(range(input_shape[2]))
        fs_ts_ids.append((t_ids, s_ids))
        feature_streams.append(feature_stream)

    # 2/3/4. Cost volume + 3D aggregation + Regression
    cv_ca_pyr_levels = pyr_levels[1:]
    for pyr_level in cv_ca_pyr_levels[::-1]:
        cv_streams = []
        scale_factor = math.pow(2, pyr_level)
        pyr_level_ndl = int(num_disp_labels / scale_factor)

        # 2. Cost volume
        for fs_id, feature_stream in enumerate(feature_streams):
            pyr_fs = [fs_ep[pyr_level-1] for fs_ep in feature_stream]
            cost_volume = Lambda(compute_cost_volume,
                                 arguments={"t_s_ids": fs_ts_ids[fs_id],
                                            "min_disp": min_disp/scale_factor,
                                            "max_disp": max_disp/scale_factor,
                                            "labels": pyr_level_ndl,
                                            "move_path": "LT"})(pyr_fs)
            cv_streams.append(cost_volume)

        # Multiple streams
        if len(cv_streams) > 1:
            cost_volume = concatenate(cv_streams)

        # 3/4. 3D aggregation + Regression
        # 3. 3D aggregation
        if pyr_level == cv_ca_pyr_levels[0]:
            ca_paras = {'ks': 3, 'stride': 2, 'padding': "same", 'filter': nb_filt1*2,
                        'activation': activ, 'conv_type': conv_type, 'n_dc': 1}
            output = cost_aggregation(cost_volume, ca_paras=ca_paras)
        else:
            ca_paras = {'ks': 3, 'stride': 2, 'padding': "same", 'filter': nb_filt1*4,
                        'activation': activ, 'conv_type': conv_type, 'n_dc': 1}
            output = cost_aggregation(cost_volume, ca_paras=ca_paras)
            output = UpSampling3D(size=(2, 2, 2), name="u_s{}".format(pyr_level))(output)

        # 4. Regression
        logger.info("=> regression at scale level {}".format(pyr_level))
        output = Lambda(lambda op: tf.nn.softmax(op, axis=1))(output)
        pl_o = Lambda(soft_min_reg,
                      arguments={"axis": 1,
                                 "min_disp": min_disp,
                                 "max_disp": max_disp,
                                 "labels": num_disp_labels},
                      name="sm_disp{}".format(pyr_level))(output)
        pyr_outputs.append(pl_o)
    d2 = Average()(pyr_outputs[:2]) # outputs at scale level 1 and 2


    ########## Branch_1: no_cv ##########
    block_n = 8 # blocks
    ifn = 40 # filter

    # Branch_1: 2D aggregation
    pl_features = []
    pl_feature_streams = []
    for x in input_layers:
        x = ReflectionPadding2D(padding=([4, 4], [4, 4]))(x)
        feature_paras = {'ks': ks, 'stride': 1, 'padding': "zero", 'filter': 1*[ifn],
                         'activation': activ, 'conv_type': conv_type, 'layer_nums': 1}
        x = cna_m(x, feature_paras, layer_names='random')
        pl_feature_streams.append(x)
    x = concatenate(pl_feature_streams) # merge layers
    pl_features.append(x)

    pyr_level = pyr_levels[0] # = 0
    fn = [i for i in block_n*[ifn*len(input_layer_names)]]
    cna_paras = {'ks': ks, 'stride': 1, 'padding': "valid", 'filter': fn,
                 'activation': activ, 'conv_type': conv_type, 'layer_nums': block_n}
    x = cna_m(pl_features[pyr_level], cna_paras, layer_names='random')
    x = conv_2d(x, num_disp_labels, ks=ks, padding="zero")

    # Branch_1: Regression
    logger.info("=> regression at scale level {}".format(pyr_level))
    x = Lambda(lambda op: tf.nn.softmax(op, axis=-1))(x)
    d1 = Lambda(soft_min_reg,
                arguments={"axis": -1,
                           "min_disp": min_disp,
                           "max_disp": max_disp,
                           "labels": num_disp_labels},
                name="sm_disp_{}".format(pyr_level))(x)


    ########## Output ##########
    d0 = Average()([d2, d1])
    output_layers.append(d2)
    output_layers.append(d1)
    output_layers.append(d0)


    manet_model = Model(inputs=input_layers,
                        outputs=output_layers)
    if config.model_infovis:
        manet_model.summary()
    return manet_model
