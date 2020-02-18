import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--image_size', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--sample_batch_size', type=int, default=1, help='sample batch size')
parser.add_argument('--num_epoch', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--epoch', type=int, default=0, help='epochs in current train')
parser.add_argument('--checkpoint_dir', default='checkpoints', help="path to saved models (to continue training)")
parser.add_argument('--sample_dir', default='samples', help='folder to output images and model checkpoints')

parser.add_argument('--workers', type=int, default=5, help='number of data loading workers')
parser.add_argument('--scale_factor', type=int, default=4, help='scale factor for super resolution')
parser.add_argument('--nf', type=int, default=32, help='number of filter in esrgan')
parser.add_argument('--b1', type=float, default=0.9,
                    help='coefficients used for computing running averages of gradient and its square')
parser.add_argument('--b2', type=float, default=0.999,
                    help="coefficients used for computing running averages of gradient and its square")
parser.add_argument('--weight_decay', type=float, default=1e-2, help='weight decay')

parser.add_argument('--p_lr', type=float, default=2e-4, help='learning rate when when training perceptual oriented')
parser.add_argument('--p_decay_batch_size', type=int, default=2e5, help='batch size where learning rate halve each '
                                                                          'when training perceptual oriented')
parser.add_argument('--p_content_loss_factor', type=float, default=1, help='content loss factor when training '
                                                                           'perceptual oriented')
parser.add_argument('--p_perceptual_loss_factor', type=float, default=0, help='perceptual loss factor when training '
                                                                              'perceptual oriented')
parser.add_argument('--p_adversarial_loss_factor', type=float, default=0, help='adversarial loss factor when '
                                                                               'training perceptual oriented')

parser.add_argument('--g_lr', type=float, default=1e-4, help='learning rate when when training generator oriented')
parser.add_argument('--g_decay_batch_size', type=int, default=1e5, help='batch size where learning rate halve each '
                                                                          'when training generator oriented')
parser.add_argument('--g_content_loss_factor', type=float, default=1e-1, help='content loss factor when training '
                                                                              'generator oriented')
parser.add_argument('--g_perceptual_loss_factor', type=float, default=1, help='perceptual loss factor when training '
                                                                              'generator oriented')
parser.add_argument('--g_adversarial_loss_factor', type=float, default=5e-3, help='adversarial loss factor when '
                                                                                  'training generator oriented')

parser.add_argument('--is_perceptual_oriented', type=bool, default=True, help='')

url = ['http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip',
       'http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar'
       ]

parser.add_argument('--dataset_url', type=list, default=url, help='the url of DIV2K dataset for super resolution')
def get_config():
    return parser.parse_args()
