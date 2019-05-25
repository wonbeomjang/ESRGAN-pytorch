import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--image_size', type=int, default=224, help='the height / width of the input image to network')
parser.add_argument('--batch_size', type=int, default=2, help='batch size')
parser.add_argument('--sample_batch_size', type=int, default=1, help='sample batch size')
parser.add_argument('--num_epoch', type=int, default=60, help='number of epochs to train for')
parser.add_argument('--epoch', type=int, default=0, help='epochs in current train')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--data_dir', default='dataset', help='path to dataset')
parser.add_argument('--checkpoint_dir', default='checkpoints', help="path to saved models (to continue training)")
parser.add_argument('--sample_dir', default='samples', help='folder to output images and model checkpoints')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--scale_factor', type=int, default=2, help='')

def get_config():
    return parser.parse_args()
