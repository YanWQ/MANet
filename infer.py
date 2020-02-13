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
from __future__ import print_function
from collections import OrderedDict
import argparse
import os
import time

########## Import third-party libs ##########
import matplotlib.pyplot as plt
import numpy as np

########## Import our libs ##########
from dataset import get_preds_data
from model import manet
from utils import *



parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--data_root', type=str, default='./Data') # YOUR_DATA_ROOT
parser.add_argument('--dataset', type=str, default='CVIA_HCI_val') # YOUR_DATA_SET.txt in YOUR_DATA_ROOT
parser.add_argument('--model_path', type=str, default='./Model') # YOUR_MODEL_PATH
parser.add_argument('--move_path', type=str, default='LT') # LT: Left-right, Top-bottom
parser.add_argument('--model_infovis', type=bool, default=True)
config = parser.parse_args()



def infer(model_weights_medium, data_for_predictions, model_pred=None, logger=None, show_time=False):

    def get_weights(model_weights_medium):
        # load model weights
        if isinstance(model_weights_medium, str):
            model_pred.load_weights(model_weights_medium)

    for data_key, data_value in data_for_predictions.items():
        imgs = data_value[0]
        get_weights(model_weights_medium)
        start = time.time()
        ########## predict ##########
        outputs = model_pred.predict(imgs, batch_size=config.batch_size)[-1]
        end = time.time()
        if show_time:
            logger.info("=> elapsed time: {}s".format(end-start))
        logger.info("=> output_tmp: {}".format(outputs.shape))

    return outputs

def run():
    ########## call logger ##########
    logger = get_logger()

    ########## choose CPU or GPU ##########
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" # "-1": cpu, "0": 'gtx1080 ti' (in our case)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable printing tensorflow gpu info etc.

    ########## prepare dataset ##########
    lf_shape = (512, 512, 9, 9, 3) # CVIA_HCI
    #lf_shape = (434, 625, 9, 9, 3) # EPFL
    config.lf_shape = lf_shape

    input_chns = 7 # input channels to MANet
    input_img_shape = [lf_shape[0], lf_shape[1]] # input image shape to MANet
    # padding
    if input_img_shape[0] % 8 == 0 and input_img_shape[0] % 8 == 0:
        config.pad = None
        input_shape = (input_img_shape[0], input_img_shape[1], input_chns)
    else:
        pad_n_hl, pad_n_hr, pad_n_wl, pad_n_wr = 0, 0, 0, 0
        if input_img_shape[0] % 8 != 0:
            pad_n_hl = int(8 - img_shape[0] % 8)/2
            pad_n_hr = (8 - img_shape[0] % 8) - pad_n_hl
        if input_img_shape[1] % 8 != 0:
            pad_n_wl = int(8 - img_shape[1] % 8)/2
            pad_n_wr = (8 - img_shape[1] % 8) - pad_n_wl
        config.pad = [pad_n_hl, pad_n_hr, pad_n_wl, pad_n_wr]
        input_shape = (input_img_shape[0]+(pad_n_hl+pad_n_hr), input_img_shape[1]+(pad_n_wl+pad_n_wr), input_chns)
    config.input_shape = input_shape

    preds_x = get_preds_data(config, logger=logger)
    data_for_predictions = OrderedDict()
    data_for_predictions = {config.dataset: [preds_x]}

    ########## prepare model ##########
    input_layer_names = ["x90d", "x0d", "x45d", "xm45d"]
    manet_model = manet(input_layer_names, input_shape, config=config, logger=logger)

    ########## start to infer ##########
    for model_id, model_weights_file in enumerate(os.listdir(config.model_path)):
        if '.hdf5' in model_weights_file:
            logger.info("load model weights {}".format(model_weights_file))
            if model_id == 0:
                # dry run
                x = np.zeros((1, 512, 512, 7), dtype=np.float32)
                infer(os.path.join(config.model_path, model_weights_file),
                      {"dry_run": [[x, x, x, x]]},
                      model_pred=manet_model,
                      logger=logger)
            # infer
            outputs = infer(os.path.join(config.model_path, model_weights_file),
                            data_for_predictions,
                            model_pred=manet_model,
                            logger=logger,
                            show_time=True)
            logger.info(outputs.shape)
            plt.imsave('./Results/example.png', outputs[0, ..., 0])



if __name__ == '__main__':
    run()
