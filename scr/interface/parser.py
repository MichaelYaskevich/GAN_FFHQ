import argparse


def make_parser():
    """
    Creates parser with all subparsers
    :return: argparse.ArgumentParser
    """

    description = """
        train: trains the gan and save training results
        eval: uses saved gan parameters to generate images and count accuracy
        To see help of subparser write 
        python main.py имя_подпарсера -h
        """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=description)

    subparsers = parser.add_subparsers(help='', dest='name')

    add_train_parser(subparsers)
    add_eval_parser(subparsers)

    return parser


def add_train_parser(subparsers):
    """Add train subpareser"""

    train_parser = subparsers.add_parser(
        'train',
        description='trains the gan and save training results such as gan parameters, '
                    'losses, scores, generated images, accuracy and tsne results plot',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    train_parser.add_argument(
        'image_size', type=int, nargs=1, choices=[64, 128, 256, 512, 1024],
        help='image size to build relevant gan architecture'
    )
    train_parser.add_argument(
        'epochs', type=int, nargs=1,
        help=f'epochs count for training')
    train_parser.add_argument(
        'lr', type=float, nargs=1, default=0.0002,
        help='learning rate for training')
    train_parser.add_argument(
        'dir_to_save', type=str, nargs=1,
        help='directory where to save results')


def add_eval_parser(subparsers):
    """Add eval subpareser"""

    eval_parser = subparsers.add_parser(
        'eval',
        description='uses saved gan parameters (for 128x128 image size) '
                    'to generate images and count accuracy',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    eval_parser.add_argument(
        'image_size', type=int, nargs=1, choices=[64, 128, 256, 512, 1024],
        help='image size to build relevant gan architecture'
    )
    eval_parser.add_argument(
        'images_count', type=int, nargs=1,
        help=f'count of generated images')
    eval_parser.add_argument(
        'dir_to_save', type=str, nargs=1,
        help='directory where to save results')
    eval_parser.add_argument(
        'path_to_weights', type=str, nargs=1,
        help='path to the weights of the gan, '
             'which is zip archive with discriminator.pt and generator.pt')
