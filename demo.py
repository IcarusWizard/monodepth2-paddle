import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import paddle
import paddle.nn.functional as F
from paddle.vision import transforms

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist
from evaluate_depth import STEREO_SCALE_FACTOR
from utils import load_weight_file


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', required=True)
    parser.add_argument('--load_weights_folder', type=str,
                        help='path to the weight files')
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")

    return parser.parse_args()


def test_simple(args):
    """
    Function to predict for a single image or folder of images
    """

    print("-> Loading model from ", args.load_weights_folder)
    encoder_path = os.path.join(args.load_weights_folder, "encoder")
    depth_decoder_path = os.path.join(args.load_weights_folder, "depth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, 'scratch')
    loaded_dict_enc = load_weight_file(encoder_path)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_dict(filtered_dict_enc)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc)

    loaded_dict = load_weight_file(depth_decoder_path)
    depth_decoder.load_dict(loaded_dict)

    depth_decoder.eval()

    # PREDICTING ON EACH IMAGE IN TURN
    with paddle.no_grad():
        # Load image and preprocess
        input_image = pil.open(args.image_path).convert('RGB')
        original_width, original_height = input_image.size
        input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)

        # PREDICTION
        features = encoder(input_image)
        outputs = depth_decoder(features)

        disp = outputs[("disp", 0)]
        disp_resized = F.interpolate(disp, (original_height, original_width), mode="bilinear", align_corners=False)

        # Saving numpy file
        output_directory = os.path.dirname(args.image_path)
        output_name = os.path.splitext(os.path.basename(args.image_path))[0]
        scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
        name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
        np.save(name_dest_npy, scaled_disp.cpu().numpy())

        # Saving colormapped depth image
        disp_resized_np = disp_resized.squeeze().cpu().numpy()
        vmax = np.percentile(disp_resized_np, 95)
        normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
        im = pil.fromarray(colormapped_im)

        name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))
        im.save(name_dest_im)

    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    test_simple(args)
