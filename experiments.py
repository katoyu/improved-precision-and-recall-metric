# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Functions to compute realism score and StyleGAN truncation sweep."""

import numpy as np
import os
import PIL.Image
from time import time

import dnnlib
from precision_recall import DistanceBlock
from precision_recall import knn_precision_recall_features
from precision_recall import ManifoldEstimator
from utils import initialize_feature_extractor
from utils import initialize_stylegan

import cv2
import glob

#----------------------------------------------------------------------------
# Helper functions.

def save_image(img_t, filename):
    t = img_t.transpose([1, 2, 0])  # [RGB, H, W] -> [H, W, RGB]
    PIL.Image.fromarray(t.astype(np.uint8), 'RGB').save(filename)

def generate_single_image(Gs, latent, truncation, fmt):
    gen_image = Gs.run(latent, None, truncation_psi=truncation, truncation_cutoff=18, randomize_noise=True, output_transform=fmt)
    gen_image = np.clip(gen_image, 0, 255).astype(np.uint8)
    return gen_image

#----------------------------------------------------------------------------

def compute_stylegan_truncation(datareader, minibatch_size, num_images, truncations,
                                num_gpus, random_seed, save_txt=None, save_path=None):
    """StyleGAN truncation sweep. (Fig. 4)

        Args:
            datareader (): FFHQ datareader object.
            minibatch_size (int): Minibatch size.
            num_images (int): Number of images used to evaluate precision and recall.
            truncations (list): List of truncation psi values.
            save_txt (string): Name of result file.
            save_path (string): Absolute path to directory where result textfile is saved.
            num_gpus (int): Number of GPUs used.
            random_seed (int): Random seed.

    """
    print('Running StyleGAN truncation sweep...')
    rnd = np.random.RandomState(random_seed)
    fmt = dict(func=dnnlib.tflib.convert_images_to_uint8)

    # Initialize VGG-16.
    feature_net = initialize_feature_extractor()

    # Initialize StyleGAN generator.
    Gs = initialize_stylegan()

    metric_results = np.zeros([len(truncations), 3], dtype=np.float32)
    for i, truncation in enumerate(truncations):
        print('Truncation %g' % truncation)
        it_start = time()

        # Calculate VGG-16 features for real images.
        print('Reading real images...')
        ref_features = np.zeros([num_images, feature_net.output_shape[1]], dtype=np.float32) # (50000,4096)
        print('ref_features', ref_features.shape, ref_features, type(ref_features))
        for begin in range(0, num_images, minibatch_size):
            end = min(begin + minibatch_size, num_images)
            real_batch, _ = datareader.get_minibatch_np(end - begin) # (8,3,n,n) n = 2^a, *_a.tfrecords
            # print('real_batch', real_batch.shape, real_batch)
            print('real_batch', real_batch.shape, type(real_batch))
            ref_features[begin:end] = feature_net.run(real_batch, num_gpus=num_gpus, assume_frozen=True)
            print('ref_features', ref_features.shape, ref_features, type(ref_features))

        # Calculate VGG-16 features for generated images.
        print('Generating images...')
        eval_features = np.zeros([num_images, feature_net.output_shape[1]], dtype=np.float32) # (50000,4096)
        print('eval_features', eval_features.shape, eval_features)
        for begin in range(0, num_images, minibatch_size):
            end = min(begin + minibatch_size, num_images)
            latent_batch = rnd.randn(end - begin, *Gs.input_shape[1:]) # (8,512)
            # print('latent_batch', latent_batch.shape, latent_batch)
            print('latent_batch', latent_batch.shape)
            gen_images = Gs.run(latent_batch, None, truncation_psi=truncation, truncation_cutoff=18, randomize_noise=True, output_transform=fmt) # (8,3,1024,1024)
            print('gen_images', gen_images.shape)
            eval_features[begin:end] = feature_net.run(gen_images, num_gpus=num_gpus, assume_frozen=True)
            print('eval_features', eval_features.shape, eval_features)

        # Calculate k-NN precision and recall.
        state = knn_precision_recall_features(ref_features, eval_features, num_gpus=num_gpus)
        print('state', state)

        # Store results.
        metric_results[i, 0] = truncation
        metric_results[i, 1] = state['precision'][0]
        metric_results[i, 2] = state['recall'][0]

        # Print progress.
        print('Precision: %0.3f' % state['precision'][0])
        print('Recall: %0.3f' % state['recall'][0])
        print('Iteration time: %gs\n' % (time() - it_start))

    # Save results.
    if save_txt:
        result_path = save_path
        result_file = os.path.join(result_path, 'stylegan_truncation.txt')
        header = 'truncation_psi,precision,recall'
        np.savetxt(result_file, metric_results, header=header,
                   delimiter=',', comments='')

#----------------------------------------------------------------------------

def compute_stylegan_realism(datareader, minibatch_size, num_images, num_gen_images,
                             show_n_images, truncation, num_gpus, random_seed,
                             save_images=False, save_path=None):
    """Calculate realism score for StyleGAN samples. (Fig. 11)
    
        Args:
            datareader (): FFHQ datareader object.
            minibatch_size (int): Minibatch size.
            num_images (int): Number of images used to evaluate precision and recall.
            num_gen_images (int): Number of generated images where low and high quality
                samples are selected.
            show_n_images (int): Number of low and high quality samples selected.
            truncation (float): Amount of truncation applied to StyleGAN.
            num_gpus (int): Number of GPUs used.
            random_seed (int): Random seed.
            save_images (bool): Save images.
            save_path (string): Absolute path to directory where result textfile is saved.

    """
    print('Running StyleGAN realism...')
    rnd = np.random.RandomState(random_seed)
    fmt = dict(func=dnnlib.tflib.convert_images_to_uint8)

    # Initialize VGG-16.
    feature_net = initialize_feature_extractor()

    # Initialize StyleGAN generator.
    Gs = initialize_stylegan()

    # Read real images.
    print('Reading real images...')
    real_features = np.zeros([num_images, feature_net.output_shape[1]], dtype=np.float32)
    print('real_features', real_features)
    for begin in range(0, num_images, minibatch_size):
        end = min(begin + minibatch_size, num_images)
        real_batch, _ = datareader.get_minibatch_np(end - begin)
        # print('real_batch', real_batch.shape, real_batch)
        print('real_batch', real_batch.shape)
        real_features[begin:end] = feature_net.run(real_batch, num_gpus=num_gpus, assume_frozen=True)

    # Estimate manifold of real images.
    print('Estimating manifold of real images...')
    distance_block = DistanceBlock(feature_net.output_shape[1], num_gpus)
    real_manifold = ManifoldEstimator(distance_block, real_features, clamp_to_percentile=50)

    # Generate images.
    print('Generating images...')
    latents = np.zeros([num_gen_images, Gs.input_shape[1]], dtype=np.float32)
    print('latent', latent)
    fake_features = np.zeros([num_gen_images, feature_net.output_shape[1]], dtype=np.float32)
    for begin in range(0, num_gen_images, minibatch_size):
        end = min(begin + minibatch_size, num_gen_images)
        latent_batch = rnd.randn(end - begin, *Gs.input_shape[1:])
        # print('latent_batch', latent_batch.shape, latent_batch)
        print('latent_batch', latent_batch.shape)
        gen_images = Gs.run(latent_batch, None, truncation_psi=truncation, truncation_cutoff=18, randomize_noise=True, output_transform=fmt)
        fake_features[begin:end] = feature_net.run(gen_images, num_gpus=num_gpus, assume_frozen=True)
        latents[begin:end] = latent_batch

    # Estimate quality of individual samples.
    _, realism_scores = real_manifold.evaluate(fake_features, return_realism=True)

    if save_images and save_path is not None:
        result_dir = os.path.join(save_path, 'stylegan_realism', 'truncation%0.2f' % truncation)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

    # Save samples with lowest and highest realism.
    num_saved = show_n_images

    # Sort realism scores.
    highest_realism_idx = realism_scores.argsort()[-num_saved:][::-1]
    lowest_realism_idx = realism_scores.argsort()[:num_saved]

    print('Saving %i low and high quality samples...' % num_saved)
    for i in range(num_saved):
        low_idx = lowest_realism_idx[i]
        high_idx = highest_realism_idx[i]

        # Get corresponding latents.
        low_quality_latent = latents[low_idx]
        high_quality_latent = latents[high_idx]

        # Generate images.
        low_quality_img = generate_single_image(Gs, low_quality_latent[None, :], truncation, fmt)[0]
        high_quality_img = generate_single_image(Gs, high_quality_latent[None, :], truncation, fmt)[0]

        if save_images:
            low_realism_score = realism_scores[low_idx]
            high_realism_score = realism_scores[high_idx]
            save_image(low_quality_img, os.path.join(result_dir, 'low_realism_%f_%i.png' % (low_realism_score, i)))
            save_image(high_quality_img, os.path.join(result_dir, 'high_realism_%f_%i.png' % (high_realism_score, i)))
        else:
            low_quality_img.show()
            high_quality_img.show()

    print('Done evaluating StyleGAN realism.\n')

#----------------------------------------------------------------------------

# katoyu coding area
#----------------------------------------------------------------------------

# def compute_stylegan_ccr(datareader, minibatch_size, num_images, ccrs,
#                                 num_gpus, random_seed, save_txt=None, save_path=None):
#     """StyleGAN ccr sweep. (Fig. 4)

#         Args:
#             datareader (): FFHQ datareader object.
#             minibatch_size (int): Minibatch size.
#             num_images (int): Number of images used to evaluate precision and recall.
#             ccrs (list): List of ccr values.
#             save_txt (string): Name of result file.
#             save_path (string): Absolute path to directory where result textfile is saved.
#             num_gpus (int): Number of GPUs used.
#             random_seed (int): Random seed.

#     """
#     print('Running StyleGAN ccr sweep...')
#     rnd = np.random.RandomState(random_seed)
#     fmt = dict(func=dnnlib.tflib.convert_images_to_uint8)

#     # Initialize VGG-16.
#     feature_net = initialize_feature_extractor()

#     # Initialize StyleGAN generator.
#     # TODO: 提案システムへ修正
#         # TODO: 学習済みモデルの読み込み
#     Gs = initialize_stylegan()

#     metric_results = np.zeros([len(ccrs), 3], dtype=np.float32)
#     for i, ccr in enumerate(ccrs):
#         print('ccr %g' % ccr)
#         it_start = time()

#         # Calculate VGG-16 features for real images.
#         print('Reading real images...')
#         ref_features = np.zeros([num_images, feature_net.output_shape[1]], dtype=np.float32)
#         for begin in range(0, num_images, minibatch_size):
#             end = min(begin + minibatch_size, num_images)

#             # TODO: tfrecordなしでのデータ読み込み
#             dataset = dataset_load(path='')
#             # TODO: tfrecordなしでのバッチ作成
#                 # TODO: real_batch.shapeを確認
            
#             real_batch, _ = datareader.get_minibatch_np(end - begin)
#             ref_features[begin:end] = feature_net.run(real_batch, num_gpus=num_gpus, assume_frozen=True)

#         # Calculate VGG-16 features for generated images.
#         print('Generating images...')
#         eval_features = np.zeros([num_images, feature_net.output_shape[1]], dtype=np.float32)
#         for begin in range(0, num_images, minibatch_size):
#             end = min(begin + minibatch_size, num_images)
#             latent_batch = rnd.randn(end - begin, *Gs.input_shape[1:])

#             # TODO: 画像生成を提案システムの方に置き換え
#                 # TODO: データセット画像の潜在空間への埋め込み
#                 # TODO: 次元圧縮
#                 # TODO: 50000枚サンプリング（暖機運転期間を設ける）→gen_imagesとして使用
#                     # TODO: gen_images.shapeを確認

#             gen_images = Gs.run(latent_batch, None, ccr=ccr, ccr_cutoff=18, randomize_noise=True, output_transform=fmt)
#             eval_features[begin:end] = feature_net.run(gen_images, num_gpus=num_gpus, assume_frozen=True)

#         # Calculate k-NN precision and recall.
#         state = knn_precision_recall_features(ref_features, eval_features, num_gpus=num_gpus)

#         # Store results.
#         metric_results[i, 0] = ccr
#         metric_results[i, 1] = state['precision'][0]
#         metric_results[i, 2] = state['recall'][0]

#         # Print progress.
#         print('Precision: %0.3f' % state['precision'][0])
#         print('Recall: %0.3f' % state['recall'][0])
#         print('Iteration time: %gs\n' % (time() - it_start))

#     # Save results.
#     if save_txt:
#         result_path = save_path
#         result_file = os.path.join(result_path, 'stylegan_ccr.txt')
#         header = 'ccr,precision,recall'
#         np.savetxt(result_file, metric_results, header=header,
#                    delimiter=',', comments='')


# def dataset_load_from_npy(path=''):
#     # load dataset from npy
#     npy_paths = glob.glob(os.path.join(path, '*.npy'))
    
#     # npy to img
#     imgs = ###
    
#     return imgs


def compute_stylegan_ccr_from_imgs(datareader, minibatch_size, num_images, ccrs,
                                num_gpus, random_seed, save_txt=None, save_path=None):
    """StyleGAN ccr sweep. (Fig. 4)

        Args:
            datareader (): FFHQ datareader object.
            minibatch_size (int): Minibatch size.
            num_images (int): Number of images used to evaluate precision and recall.
            ccrs (list): List of ccr values.
            save_txt (string): Name of result file.
            save_path (string): Absolute path to directory where result textfile is saved.
            num_gpus (int): Number of GPUs used.
            random_seed (int): Random seed.

    """
    print('Running StyleGAN ccr sweep...')
    rnd = np.random.RandomState(random_seed)
    fmt = dict(func=dnnlib.tflib.convert_images_to_uint8)

    # Initialize VGG-16.
    feature_net = initialize_feature_extractor()

    # Initialize StyleGAN generator.
    Gs = initialize_stylegan()

    metric_results = np.zeros([len(ccrs), 3], dtype=np.float32)
    for i, ccr in enumerate(ccrs):
        print('ccr %g' % ccr)
        it_start = time()

        # Calculate VGG-16 features for real images.
        print('Reading real images...')
        ref_features = np.zeros([num_images, feature_net.output_shape[1]], dtype=np.float32)
        for begin in range(0, num_images, minibatch_size):
            end = min(begin + minibatch_size, num_images)

            # TODO: tfrecordなしでのデータ読み込み
            dataset = dataset_load_from_img(path='/content/drive/My Drive/stylegan2encoder/out/Johnnys103/face-1024x1024/imgs/')
            # TODO: tfrecordなしでのバッチ作成
                # TODO: real_batch.shapeを確認  # (8,3,n,n) n = 2^a, *_a.tfrecords
            
            real_batch, _ = datareader.get_minibatch_np(end - begin)
            ref_features[begin:end] = feature_net.run(real_batch, num_gpus=num_gpus, assume_frozen=True)

        # Calculate VGG-16 features for generated images.
        print('Generating images...')
        eval_features = np.zeros([num_images, feature_net.output_shape[1]], dtype=np.float32)

        # TODO: 生成済み画像を使用、2048枚読み込み
        gen_images = dataset_load(path='/content/drive/My Drive/stylegan2encoder/out/rizin112/face-1024x1024/2048/imgs/ccr' + f'{ccr:.02f}')

        for begin in range(0, num_images, minibatch_size):
            end = min(begin + minibatch_size, num_images)
            latent_batch = rnd.randn(end - begin, *Gs.input_shape[1:])
            
            print(begin, end)
            # gen_images = Gs.run(latent_batch, None, ccr=ccr, ccr_cutoff=18, randomize_noise=True, output_transform=fmt) # (batch_size,3,1024,1024)
            gen_images_batch = gen_images[begin:end] # (n,3,1024,1024) -> (batch_size,3,1024,1024)
            
            eval_features[begin:end] = feature_net.run(gen_images, num_gpus=num_gpus, assume_frozen=True)

        # Calculate k-NN precision and recall.
        state = knn_precision_recall_features(ref_features, eval_features, num_gpus=num_gpus)

        # Store results.
        metric_results[i, 0] = ccr
        metric_results[i, 1] = state['precision'][0]
        metric_results[i, 2] = state['recall'][0]

        # Print progress.
        print('Precision: %0.3f' % state['precision'][0])
        print('Recall: %0.3f' % state['recall'][0])
        print('Iteration time: %gs\n' % (time() - it_start))

    # Save results.
    if save_txt:
        result_path = save_path
        result_file = os.path.join(result_path, 'stylegan_ccr.txt')
        header = 'ccr,precision,recall'
        np.savetxt(result_file, metric_results, header=header,
                   delimiter=',', comments='')


def dataset_load_from_img(path_img=''):
    # load dataset from npy
    paths = glob.glob(os.path.join(path_img, '*.png'))
    paths = sorted(paths)
    
    imgs = []
    for path in paths:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose([2, 0, 1])
        imgs.append(img)

    imgs_np = np.array(imgs)
    print(imgs_np.shape)
    return imgs_np 