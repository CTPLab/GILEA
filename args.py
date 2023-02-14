import random

from cfgs import Config
from pathlib import Path
from argparse import ArgumentParser


def _get_pths(args):
    if args.is_total:
        args.ckpt_path = [args.ckpt_path /
                          'checkpoints' / f'iteration_{args.n_iter}.pt']
    else:
        # single stain protocol (channel) analysis
        # only available for rxrx19a/b datasets
        assert args.data_name != 'ham10k' and args.decoder == 'style2'
        _ckpt = str(args.ckpt_path)
        args.ckpt_path = [Path(_ckpt + f'_chn{chn}') /
                          'checkpoints' / f'iteration_{args.n_iter}.pt' for chn in range(args.img_chn)]

    save_data = args.data_name if args.data_name == 'ham10k' else args.data_cell
    save_data += f'_{args.data_splt}'
    save_info = f'{args.encoder}_{args.decoder}_{args.n_iter}_'
    save_info += f'{args.is_total}_{args.is_merge}_{args.is_layer}_{args.seed}_{args.stat_res}'

    args.save_path = args.save_path / save_data / save_info
    args.save_path.mkdir(parents=True, exist_ok=True)


def _get_dims(args):
    # the number of input channels
    args.inp_chn = args.img_chn if args.is_total else 1
    args.lay_num = 12 if args.decoder == 'style2' else 16
    args.lay_dim = 512
    args.lay_dup = 1
    # args.lay_dim = 512 if args.is_total else 128
    # args.lay_dup = 1 if args.is_total else 4
    args.cod_dim = args.lay_dim * args.lay_num

    if not args.is_total and args.is_merge:
        assert args.data_name != 'ham10k'
        args.cod_dim *= args.img_chn

    if args.is_layer:
        args.cod_dim //= args.lay_num


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='global seed (for weight initialization, data sampling, etc.). '
                        'If not specified it will be randomized (and printed on the log)')
    parser.add_argument('--task',
                        type=str,
                        help='either calculate the stats or output image plots, video demos.')

    parser.add_argument('--decoder',
                        type=str,
                        default='style2',
                        choices=('style2', 'style3'))
    parser.add_argument('--encoder',
                        type=str,
                        default='psp',
                        choices=('psp', 'e4e'))

    parser.add_argument('--is_total',
                        action='store_true',
                        help='whether load the single model trained on images with total amount of channels'
                        'or multiple models individually trained on images with each single channel')
    parser.add_argument('--is_merge',
                        action='store_true',
                        help='(Deprecated) whether to concatenate the elments if the codes are stored as list')
    parser.add_argument('--is_layer',
                        action='store_true',
                        help='(Deprecated) whether to split the codes layer-wise')

    parser.add_argument('--data_name',
                        type=str,
                        choices=('rxrx19a', 'rxrx19b', 'ham10k'),
                        help='the name of biomedical datasets used for application studies')
    parser.add_argument('--data_cell',
                        type=str,
                        choices=('VERO', 'HRCE', 'HUVEC'),
                        help='the cell types of rxrx19 datasets, VERO and HRCE are in 19a, HUVEC is in 19b.')
    parser.add_argument('--data_splt',
                        type=str,
                        help='the name of application studies used for retrieving the sub-collections of data entries')
    parser.add_argument('--data_path',
                        type=Path,
                        help='path to the data root.')

    parser.add_argument('--ckpt_path',
                        type=Path,
                        help='path to the checkpoint of the auto-encoder')
    parser.add_argument('--save_path',
                        type=Path,
                        help='path to output stats (*.npy), plots (*.png) and demos (*.mp4)')

    parser.add_argument('--n_iter',
                        type=int,
                        help='the training iterations of the checkpoint')
    parser.add_argument('--n_epoh',
                        type=int,
                        default=1,
                        help='the amount of epochs, mostly set to be 1 except for toy experiments(=4)')
    parser.add_argument('--n_eval',
                        type=int,
                        default=8,
                        help='the batch size during evaluation')
    parser.add_argument('--n_work',
                        type=int,
                        default=8,
                        help='the amount of data loader workers')

    parser.add_argument('--stat_dec',
                        action='store_true',
                        help='whether feed the latent codes to decoder (save inference time if not)')
    parser.add_argument('--stat_res',
                        action='store_true',
                        help='whether only compute the residual latent codes')
    parser.add_argument('--stat_eig',
                        type=str,
                        choices=['scm', 'cov'],
                        help='Compute the eigenvalue/vector of sample covariance matrix (scm) or covariance matrix (cov)')
    parser.add_argument('--stat_top',
                        type=int,
                        help='The amount of largest eigenvalues to be calculated')

    # the parameters for plots reported in the paper
    parser.add_argument('--plot_val',
                        action='store_true',
                        help='whether to manipulate the eigenvalue')
    parser.add_argument('--plot_vec',
                        action='store_true',
                        help='whether to manipulate the eigenvec')
    parser.add_argument('--plot_powr',
                        type=float,
                        default=2,
                        help='the power range of exponential weights multiplied by the eigenvalue (vector), '
                        'along which we manipulate the image')
    parser.add_argument('--plot_step',
                        type=int,
                        default=5,
                        help='the power step of exponential weights multiplied by the eigenvalue (vector), '
                        'along which we manipulate the image')
    parser.add_argument('--plot_base',
                        type=float,
                        default=2,
                        help='the base of exponential weights multiplied by the eigenvalue (vector), '
                        'along which we manipulate the image')
    parser.add_argument('--plot_axis',
                        type=int,
                        default=0,
                        help='the axis of the eigenvalue, along which we create the image manipulation')
    parser.add_argument('--plot_seed',
                        type=int,
                        default=0,
                        help='the seed for reproducing plot results')

    # the parameters for video demos
    parser.add_argument('--demo_val',
                        action='store_true',
                        help='whether to manipulate the eigenvalue')
    parser.add_argument('--demo_vec',
                        action='store_true',
                        help='whether to manipulate the eigenvec')
    parser.add_argument('--demo_powr',
                        type=float,
                        default=1.5,
                        help='the power range of exponential weights multiplied by the eigenvalue (vector), '
                        'along which we manipulate the video demo')
    parser.add_argument('--demo_step',
                        type=int,
                        default=61,
                        help='the power step of exponential weights multiplied by the eigenvalue (vector), '
                        'along which we manipulate the video demo')
    parser.add_argument('--demo_base',
                        type=float,
                        default=2,
                        help='the base of exponential weights multiplied by the eigenvalue (vector), '
                        'along which we manipulate the video demo')
    parser.add_argument('--demo_axis',
                        type=int,
                        default=0,
                        help='the axis of the eigenvalue, along which we create the video demo manipulation')
    parser.add_argument('--demo_cell',
                        type=int,
                        default=32,
                        help='the amount of the cells manipulated in the video demos')
    parser.add_argument('--demo_seed',
                        type=int,
                        default=0,
                        help='the seed for reproducing demo results')

    args = parser.parse_args()

    assert args.save_path is not None and \
        args.ckpt_path is not None and \
        args.data_path is not None
    if args.seed is None:
        args.seed = random.randint(0, 10 ** 9)

    if 'rxrx19' in args.data_name:
        if args.data_name == 'rxrx19b':
            assert args.data_cell == 'HUVEC'
        else:
            assert args.data_cell in ('VERO', 'HRCE')

        _dt_nm = f'{args.data_name}_{args.data_cell}_cell'
        if 'abl' in args.data_splt:
            _dt_nm += f'_{args.data_splt}'
        args.data_path = args.data_path / _dt_nm
        args.size_bat = 1

        rxrx_cfg = Config().rxrx19[args.data_cell]
        args.control = rxrx_cfg['control']
        args.img_num = rxrx_cfg['cell_num']
        args.img_chn = rxrx_cfg['cell_chn']
        args.img_dim = rxrx_cfg['cell_dim']
        args.img_buf = rxrx_cfg['cell_buf']
        args.img_large = rxrx_cfg['size']
        args.img_small = rxrx_cfg['crop']

    elif args.data_name == 'ham10k':
        args.data_path = args.data_path / 'ham10k_tiny'
        args.size_bat = args.n_eval
        args.img_chn = 3

    _get_dims(args)
    _get_pths(args)

    return args
