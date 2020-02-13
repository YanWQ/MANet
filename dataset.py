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
import os

########## Import third-party libs ##########
import numpy as np
import cv2



########## light field camera/micro-lens array IDs ##########
def get_lf_ca(config=None):
    _, _, l_t, l_s, _ = config.lf_shape
    dataset_view_nums = l_t * l_s
    ca = np.arange(dataset_view_nums)
    move_path = config.move_path
    if move_path == "LT":
        ca = np.reshape(ca, newshape=(1, dataset_view_nums))
    elif move_path == "RT":
        ca = np.reshape(np.fliplr(np.reshape(ca, newshape=(l_t, l_s))), newshape=(1, dataset_view_nums))
    elif move_path == "LD":
        ca = np.reshape(np.flipud(np.reshape(ca, newshape=(l_t, l_s))), newshape=(1, dataset_view_nums))
    return ca

########## light field scene path list ##########
def read_lf_scene_path_list(data_root='', dataset_name='', logger=None):
    lf_dir = os.path.abspath(os.getcwd())
    lf_list = ''
    with open('{}/{}.txt'.format(data_root, dataset_name)) as f:
        logger.info("Loading data from {}.txt".format(dataset_name))
        lines = f.read().splitlines()
        for line_cnt, line in enumerate(lines):
            if line != '':
                if (line_cnt + 1) == len(lines):
                    lf_list += os.path.join(lf_dir, line)
                else:
                    lf_list += os.path.join(lf_dir, line) + ' '
            logger.info('Scene: {}'.format(line))
    return lf_list.split(' ')

########## load light field images ##########
def load_lf_images(frame_paths, ca, color_space, dataset_img_shape):

    _, _, l_t, l_s, _ = dataset_img_shape

    lf_img = np.zeros(((len(frame_paths),) + dataset_img_shape[:-1]), np.uint8)

    dataset_view_nums = l_t * l_s
    scene_id = 0

    # a frame means a scene
    for frame_path in frame_paths:
        # load images
        # cam_id is a coordinate in LT (origin) system
        for cam_id in range(dataset_view_nums):
            # cam_map_id: camera mapping id (used for capturing paths)
            cam_map_id = ca[0, cam_id]
            if color_space == "gray":
                try:
                    tmp = np.float32(cv2.imread(os.path.join(frame_path, 'input_Cam0%.2d.png' % cam_map_id), 0))
                except:
                    print(os.path.join(frame_path, 'input_Cam0%.2d.png..does not exist' % cam_map_id))
                lf_img[scene_id, :, :, cam_id // l_s, cam_id - l_t * (cam_id // l_s)] = tmp
            del tmp

        scene_id = scene_id + 1
    return lf_img

########## load light field data ##########
def load_lf_data(config, color_space=None, frame_paths=None, logger=None):
    if frame_paths is None:
        frame_paths = read_lf_scene_path_list(data_root=config.data_root,
                                              dataset_name=config.dataset,
                                              logger=logger)
    # light field camera/micro-lens array IDs/NOs
    ca = get_lf_ca(config)
    # load light field images
    infer_imgs = load_lf_images(frame_paths, ca, color_space, config.lf_shape)

    return infer_imgs

########## prepare preds data ##########
def prepare_preds_data(lf_imgs_data, config=None, logger=None):
    B, H, W, T, S = lf_imgs_data.shape
    assert T == S

    preds_crop_seqs = [i for i in range(1, config.input_shape[-1]+1)]
    crop_seqs = np.array(preds_crop_seqs) # np

    scene_nums = B # number of scenes
    # spatial coordinate of central view
    stride_v = H
    stride_u = W
    # angular coordinate of central view
    l_t = crop_seqs[int((len(crop_seqs)-1)/2)]
    l_s = crop_seqs[int((len(crop_seqs)-1)/2)]
    if logger is not None:
        logger.info("Central view {},{}".format(l_t, l_s))

    x_shape = (scene_nums, stride_v, stride_u, config.input_shape[-1])
    x90d = np.zeros(x_shape, dtype=np.float32)
    x0d = np.zeros(x_shape, dtype=np.float32)
    x45d = np.zeros(x_shape, dtype=np.float32)
    xm45d = np.zeros(x_shape, dtype=np.float32)

    start1 = crop_seqs[0]
    end1 = crop_seqs[-1]
    x90d_t = preds_crop_seqs[::-1]
    x0d_s = preds_crop_seqs
    for scene_id in range(scene_nums):
        for v in range(0, 1):
            for u in range(0, 1):
                x90d[scene_id, v:v + stride_v, u:u + stride_u, :] = \
                    np.moveaxis(lf_imgs_data[scene_id, v:v + stride_v, u:u + stride_u, x90d_t, l_s], 0, -1).astype('float32')
                x0d[scene_id, v:v + stride_v, u:u + stride_u, :] = \
                    np.moveaxis(lf_imgs_data[scene_id, v:v + stride_v, u:u + stride_u, l_t, x0d_s], 0, -1).astype('float32')
                for kkk in range(start1, end1 + 1):
                    x45d[scene_id, v:v + stride_v, u:u + stride_u, int((kkk - start1))] = lf_imgs_data[scene_id,
                                                                                   v:v + stride_v,
                                                                                   u:u + stride_u,
                                                                                   end1 + start1 - kkk,
                                                                                   kkk].astype('float32')
                    xm45d[scene_id, v:v + stride_v, u:u + stride_u, int((kkk - start1))] = lf_imgs_data[scene_id,
                                                                                    v:v + stride_v,
                                                                                    u:u + stride_u,
                                                                                    kkk, kkk].astype('float32')

    if config.pad is not None:
        pad_n_hl, pad_n_hr = config.pad[:2]
        pad_n_wl, pad_n_wr = config.pad[2:]
        x90d = np.pad(x90d, ((0, 0), (pad_n_hl, pad_n_hr), (pad_n_wl, pad_n_wr), (0, 0)), mode='reflect')
        x0d = np.pad(x0d, ((0, 0), (pad_n_hl, pad_n_hr), (pad_n_wl, pad_n_wr), (0, 0)), mode='reflect')
        x45d = np.pad(x45d, ((0, 0), (pad_n_hl, pad_n_hr), (pad_n_wl, pad_n_wr), (0, 0)), mode='reflect')
        xm45d = np.pad(xm45d, ((0, 0), (pad_n_hl, pad_n_hr), (pad_n_wl, pad_n_wr), (0, 0)), mode='reflect')

    x90d = np.float32((1 / 255) * x90d)
    x0d = np.float32((1 / 255) * x0d)
    x45d = np.float32((1 / 255) * x45d)
    xm45d = np.float32((1 / 255) * xm45d)
    return [x90d, x0d, x45d, xm45d]

########## get prediction data ##########
def get_preds_data(config, logger=None):
    preds_imgs_data = load_lf_data(config,
                                   color_space="gray",
                                   logger=logger)

    preds_x = prepare_preds_data(preds_imgs_data,
                                 config=config,
                                 logger=logger)
    return preds_x
