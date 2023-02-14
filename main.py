import os
import sys
import json
import torch
import random
import logging
import numpy as np
import torchvision.transforms as transforms


from pathlib import Path
from argparse import Namespace
from wilds.common.data_loaders import get_eval_loader, get_train_loader


from args import parse_args
from style2.models.psp import pSp as psp2
from style2.models.e4e import e4e as e4e2
from style3.inversion.models.e4e3 import e4e as e4e3
from style3.inversion.models.psp3 import pSp as psp3
from style3.inversion.options.train_options import TrainOptions
from style3.inversion.options.e4e_train_options import e4eTrainOptions

from utils.stat import prep_met
from utils.common import prep_dose
from utils.task import run_all_strat, run_all_stat, run_strat_sum, run_all_vis, run_all_demo
from utils.plot import get_base_plot, get_cluster_plot, get_drug_plot, get_met_res, get_dis_res, set_vis_dict


sys.path.append('.')


def worker_init_fn(worker_id):
    np.random.seed(random.randint(0, 10 ** 9) + worker_id)


def setup_logg(args):
    """ configure the logging document that records the
    critical information during evaluation

    Args:
        args: arguments that are implemented in args.py file
            such as data_name, data_splt.
    """

    head = '{asctime}:{levelname}: {message}'
    handlers = [logging.StreamHandler(sys.stderr)]
    handlers.append(logging.FileHandler(str(args.save_path / f'log_{args.task}_{args.data_splt}_{args.stat_eig}'),
                                        mode='w'))
    logging.basicConfig(level=logging.INFO,
                        format=head,
                        style='{', handlers=handlers)
    logging.info(f'Start with arguments {args}')


def setup_seed(seed):
    """
    Args:
        seed: the seed for reproducible randomization.
    """

    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_model(ckpt_pth, name_dec, name_enc):
    """ Get the trained model with the configurations
    specified by the checkpoint, decoder and encoder name

    Args:
        ckpt_pth: path to the checkpoint storing
            important configurations
        name_dec: the name of the decoder
        name_end: the name of the encoder
    """

    ckpt = torch.load(ckpt_pth, map_location='cpu')
    opts = ckpt['opts']
    opts.update({'checkpoint_path': ckpt_pth})

    if name_dec.lower() == 'style2':
        if 'input_ch' not in opts:
            # this key (for RxRx19b single chn reconstruction)
            # does not exist in the previous version of
            # restyle encoder, add it manually here if
            # load reconstruction model trained for all chns
            opts.update({'input_ch': -1})
        opts = Namespace(**opts)
        if name_enc.lower() == 'psp':
            model = psp2(opts)
        elif name_enc.lower() == 'e4e':
            model = e4e2(opts)
    elif name_dec.lower() == 'style3':
        if name_enc.lower() == 'psp':
            opts = TrainOptions(**opts)
            model = psp3(opts)
        elif name_enc.lower() == 'e4e':
            opts = e4eTrainOptions(**opts)
            model = e4e3(opts)
    model.eval().cuda()
    del ckpt
    logging.info(f'{name_dec}_{name_enc} opts:\n{str(opts)}')
    return model


def get_avgim(model, name_dec):
    """ Get the average image that is appended
    to the input image of StyleGAN2/3 decoders.

    Args:
        model: the trained auto-encoder
        name_dec: the name of the decoder
    """

    param = {'input_code': True,
             'return_latents': False}
    if name_dec.lower() == 'style2':
        # model.latent_avg = model.decoder.mean_latent(int(1e5))[0]
        param['average_code'] = True
        # this param can be critical
        # for reducing the variance with
        # exclusion of random noise injection
        # to the average image appended to true input
        param['randomize_noise'] = False
    elif name_dec.lower() == 'style3':
        model.latent_avg = model.latent_avg.repeat(16, 1)

    model.latent_avg = model.latent_avg.unsqueeze(0)
    model.latent_avg = model.latent_avg.cuda().detach()
    avgim = model(model.latent_avg,
                  **param)[0]
    avgim = avgim.float()
    return avgim


def get_codnm(args, cod_num):
    """ Get the codes name list.

    Args:
        mult: either load the model trained on multiple (all) channels
            or models individually trained on single channel images
        gpcoder: the format of grouping codes either as a whole or layer-wise
        decoder: the name of the decoder, the amout of layers is 16 for StyleGAN3
            decoder and 12 for StyleGAN2 decoder
    """
    if args.is_total:
        assert cod_num == 1
        tsr_key = ['codes0', ]
    else:
        tsr_key = [f'codes{c}' for c in range(cod_num)]

    if args.is_merge:
        tsr_key = ['codes0', ]

    if args.is_layer:
        for tid, tsr in enumerate(tsr_key):
            tsr_key[tid] = [tsr + f'_{l}' for l in range(args.lay_num)]
        tsr_key = sum(tsr_key, [])

    return tsr_key


def get_loader(args,
               data,
               cond):
    """ Get the data loader for the follow-up stat computation
    with collecting additional info.

    Args:
        args: arguments that are implemented in args.py file
            such as data_name, data_splt.
        stat: the dictionary storing prior info for calculating
            the accumulative statistics
        data: the instantiation of wilds data class passing to the data_loader
        cond: the name of the sub-collection of data e.g., 'nv', 'mel' or 'healthy', 'severe'.
    """
    if 'nv_' in cond or 'mel_' in cond:
        if 'nv_' in cond:
            coef = [1 - float(cond[-3:]), 1]
        elif 'mel_' in cond:
            coef = [1, 1 + float(cond[-3:])]
        jitter = transforms.ColorJitter(coef, coef, coef, float(cond[-3:]))
        jitter = transforms.Compose([jitter])
        subset = data.get_subset(cond,
                                 transform=jitter)
    else:
        subset = data.get_subset(cond)

    if args.task == 'stat' and args.data_splt != 'visual':
        # fix the eval order
        data_loader = get_eval_loader
    elif args.task == 'demo':
        data_loader = get_eval_loader
    else:
        data_loader = get_train_loader

    dload = data_loader('standard',
                        subset,
                        args.size_bat,
                        **{'drop_last': False,
                           'num_workers': args.n_work,
                           'pin_memory': True})
    return dload


def get_data(args):
    if 'rxrx19' in args.data_name:
        from Dataset.rxrx19 import rxrx19Dataset
        POLT_DCT = {'GS-441524': ['royalblue', 'o', '-', [0.25, 0.41, 0.88]],
                    'Remdesivir (GS-5734)': ['royalblue', 'v', '--', [0.25, 0.41, 0.88]],
                    'Chloroquine': ['r', 's', '--', [1, 0, 0]],
                    'Hydroxychloroquine Sulfate': ['r', '^', ':', [1, 0, 0]],
                    'Idelalisib': ['r', '^', ':', [1, 0, 0]],
                    'Bortezomib': ['r', '^', ':', [1, 0, 0]],
                    'Crizotinib': ['darkgreen', 'o', ':', [0, 0.2, 0.13]],
                    'Golvatinib': ['darkgreen', 'P', '-', [0, 0.2, 0.13]],
                    'Cabozantinib': ['darkgreen', 'X', '-.', [0, 0.2, 0.13]],
                    'Mock': ['black', 'o', [1, 1, 1]], 'healthy': ['black', 'o', [1, 1, 1]], 'Irradiated': ['b', 'o', [0, 0, 1]],
                    'Infected': ['orangered', 's', [1., 0.27, 0.]], 'storm-severe': ['orangered', 's', [1., 0.27, 0.]]}
        data = rxrx19Dataset(args.control, args.data_cell,
                             args.img_num, args.img_chn,
                             args.data_path, args.data_splt,
                             args.seed)
        with open(f'Dataset/doc/{args.data_cell}_{args.data_splt}.json', 'r') as file:
            cond_dct = json.load(file)
        with open(f'Dataset/doc/{args.data_cell}.json', 'r') as file:
            hit_dct = json.load(file)
    elif args.data_name == 'ham10k':
        from Dataset.ham10k import ham10kDataset
        POLT_DCT = {'bcc': ['orangered', 's', '--', [1., 0.27, 0.]],
                    'bkl': ['darkorange', '^', ':', [1., 0.55, 0.]],
                    'mel': ['r', 'o', '-', [1, 0, 0]]}
        data = ham10kDataset(args.data_path, args.data_splt)
        cond_dct, hit_dct = data._split_dict, None
    return data, POLT_DCT, cond_dct, hit_dct


def main_stat(args, data, cond_dct,
              model, avgim, keynm):
    # if args.data_name == 'ham10k':
    #     cond_dct = data._split_tot
    # elif 'rxrx19' in args.data_name:
    #     with open(f'Dataset/doc/{args.data_cell}_{args.data_splt}.json', 'r') as file:
    #         cond_dct = json.load(file)

    for cond, cval in cond_dct.items():
        print(cond)
        stat_pth = args.save_path / cond
        stat_pth.mkdir(parents=True, exist_ok=True)

        if args.data_name == 'ham10k' or args.data_splt not in ('strat', 'abl', 'abl0', 'visual'):
            # ham10k: Since each category of ham10k has dif amount of images,
            # we use the amount as the index.
            # rxrx19: the value of each drug is [index, img_num, cel_num]
            # hence, we should assign cval[2]
            total = cval[2] if args.data_name != 'ham10k' else cval
            dload = get_loader(args, data, cond)
            run_all_stat(args,
                         stat_pth / '.pt',
                         total, dload,
                         model, avgim, keynm, True)
        else:
            # prepare accumulated dicts
            size_strat, stat_strat, met_strat = dict(), dict(), dict()
            for exp in ('1', '2'):
                size_strat[exp] = 0
                stat_strat[exp] = dict()
                stat_strat[exp]['mean'] = [0 for _ in range(len(keynm['cod']))]
                stat_strat[exp]['scm'] = [0 for _ in range(len(keynm['cod']))]
                met_strat[exp] = prep_met(met_key=keynm['met'],
                                          met_chn=args.img_chn)[1]
            # get all the doses for 1 or 2 experiments
            dose = prep_dose(cval, ('1', '2'))
            print(cond, dose)

            for dos in dose:
                size_dose, stat_dose, met_dose = dict(), dict(), dict()
                for exp in size_strat:
                    if cval[exp] is not None and dos in cval[exp]:
                        cond_dose = f'{cond}-{exp}'
                        if cond not in args.control:
                            cond_dose += f'-{dos}'
                        path = stat_pth / f'{cond_dose}_.pt'
                        size_dose[exp] = cval[exp][dos][2]
                        dload = get_loader(args, data, cond_dose)
                        stat_dose[exp], met_dose[exp] = run_all_stat(args, path,
                                                                     size_dose[exp], dload,
                                                                     model, avgim, keynm,
                                                                     strat=True)
                        # accumulate the dose for each exp
                        # re-compute eigval for drugs with one dose
                        # dos == dose[-1] is valid because if one drug exist in both exps,
                        # then the doses are the same for both exps. No corner case will
                        # break the condition dos == dose[-1]
                        if cond not in args.control and dos == dose[-1]:
                            path_exp = stat_pth / f'{cond}-{exp}_.pt'
                        else:
                            path_exp = None
                        size_strat[exp], stat_strat[exp], met_strat[exp] = run_strat_sum(size_strat[exp], size_dose[exp],
                                                                                         stat_strat[exp], stat_dose[exp],
                                                                                         met_strat[exp], met_dose[exp],
                                                                                         args.stat_top, path_exp)
                        print(cond, exp, dos,
                              size_dose[exp], size_strat[exp], '\n')

                # if the dose exists in both 1 and 2 exp, then
                # add them and obtain eigenvalues
                if cval['1'] is not None and cval['2'] is not None:
                    assert '1' in size_dose and '2' in size_dose
                    path_dose = cond if cond in args.control else f'{cond}-{dos}'
                    run_strat_sum(size_dose['1'], size_dose['2'],
                                  stat_dose['1'], stat_dose['2'],
                                  met_dose['1'], met_dose['2'],
                                  args.stat_top,
                                  stat_pth / f'{path_dose}_.pt')
                    print('merge dose:', cond, exp, dos,
                          size_dose['1'] + size_dose['2'], '\n')

            # if both exp exists, then add them and obtain eigenvalues
            # re-compute eigval for drugs with one dose
            if size_strat['1'] != 0 and size_strat['2'] != 0 and cond not in args.control:
                run_strat_sum(size_strat['1'], size_strat['2'],
                              stat_strat['1'], stat_strat['2'],
                              met_strat['1'], met_strat['2'],
                              args.stat_top,
                              stat_pth / f'{cond}_.pt')
                print('merge all:', cond, exp, dos,
                      size_strat['1'] + size_strat['2'])


def main_error(args, cond_dct):
    get_met_res(args, logging,
                ['psnr', 'ssim'], cond_dct)


def main_quant(args, plot_dct, cond_dct, hit_dct):
    if 'rxrx19' in args.data_name:
        assert args.data_splt == 'abl'

        out_dct = {}
        for i in range(1, 5):
            path = str(args.save_path).replace('ldim_1', f'ldim_{i}')
            path = path.replace('_1_True', f'_{i}_True')
            out_dct[i] = run_all_strat(args, Path(path), cond_dct, hit_dct)[0]

        # calc drug plots
        if args.data_cell == 'VERO':
            drugs = ('GS-441524', 'Remdesivir (GS-5734)',
                     'Chloroquine', 'Hydroxychloroquine Sulfate')
        elif args.data_cell == 'HRCE':
            drugs = ('GS-441524', 'Remdesivir (GS-5734)',
                     'Chloroquine', 'Hydroxychloroquine Sulfate')
        elif args.data_cell == 'HUVEC':
            drugs = ('Crizotinib', 'Golvatinib', 'Cabozantinib')
        stat, exp, tot = 'scm', 'all', 'one_dos'
        out_plot, has_drug = dict(), False
        for cond, cval in cond_dct.items():
            if cond in out_dct[1][stat][exp][tot]:
                if cond in drugs:
                    out_plot[cond], has_drug = dict(), True
                    for dos in out_dct[1][stat][exp][tot][cond]:
                        all_res = np.asarray([out_dct[i][stat][exp][tot][cond][dos]
                                              for i in range(1, 5)])
                        avg = np.mean(all_res, axis=0)
                        std = np.std(all_res, axis=0)
                        print(cond, dos, avg.shape, std.shape)
                        out_plot[cond][dos] = np.stack((avg, avg - std, avg + std),
                                                       axis=-1)
                elif cond in args.control:
                    all_res = np.asarray([out_dct[i][stat][exp][tot][cond]
                                          for i in range(1, 5)])
                    avg = np.mean(all_res, axis=0)
                    std = np.std(all_res, axis=0)
                    print(cond, avg.shape, std.shape)
                    out_plot[cond] = np.stack((avg, avg - std, avg + std),
                                              axis=-1)

        if out_plot and has_drug:
            path = args.save_path / 'main_quant' / \
                f'drug_{stat}-{exp}-{tot}'
            path.mkdir(parents=True, exist_ok=True)
            get_drug_plot(args, path, plot_dct, out_plot)
    elif args.data_name == 'ham10k':
        get_dis_res(args, logging, plot_dct)


def main_visual(args, plot_dct, model):
    if args.data_name == 'ham10k':
        plot_dct.update({'nv': ['b', 'o', '--', [0, 0, 1]]})
        # cmp_lst = ['nv', 'bcc']
        # cmp_lst = ['nv', 'bkl']
        cmp_lst = ['nv', 'mel']
    elif args.data_name == 'rxrx19a':
        neg_drg = 'Bortezomib' if args.data_cell == 'HRCE' else 'Idelalisib'
        # cmp_lst = ['Mock', 'Infected', neg_drg]
        cmp_lst = ['Mock', 'Infected', 'Remdesivir (GS-5734)']
    elif args.data_name == 'rxrx19b':
        cmp_lst = ['healthy', 'storm-severe', 'Bortezomib']
        cmp_lst = ['healthy', 'storm-severe', 'Crizotinib']

    eig_dct = {}
    for cid, cmp in enumerate(cmp_lst):
        prefix = ''
        # add 'best_' to drugs
        if 'rxrx19' in args.data_name and cid >= 2:
            prefix = 'best_'
        eig_dct.update({cmp: set_vis_dict(args.save_path / cmp, prefix)})
    run_all_vis(args, 5,
                eig_dct, plot_dct,
                model, 'PC')


def main_demo(args, data,
              plot_dct, model, avgim):
    if args.data_name == 'ham10k':
        plot_dct.update({'nv': ['b', 'o', '--', [0, 0, 1]]})
        cmp_lst = ['nv', 'mel']
    elif args.data_name == 'rxrx19a':
        cmp_lst = ['Mock', 'Infected']
        # cmp_lst = ['Infected', 'Remdesivir (GS-5734)']
    elif args.data_name == 'rxrx19b':
        cmp_lst = ['healthy', 'storm-severe']
    ref, cnd = cmp_lst[0], cmp_lst[1]

    stat_dct = {cmp: set_vis_dict(args.save_path / cmp) for cmp in cmp_lst}
    load_dct = {cmp: get_loader(args, data, cmp) for cmp in cmp_lst}
    setup_seed(args.seed)
    run_all_demo(args, ref, cnd,
                 stat_dct,
                 plot_dct[cnd][-1],
                 'utils/demo_helper/arial.ttf',
                 load_dct[cnd], model, avgim)


def main_base(args, cond_dct, hit_dct):
    out_dct, cmp_dct = run_all_strat(args,
                                     args.save_path,
                                     cond_dct, hit_dct)

    # calc topk plots
    for stat in ('scm', ):
        for exp in ('all',):
            for tot in ('max_dos', 'all_dos'):
                out_plot, hit_plot = dict(), dict()
                for cond, cval in cond_dct.items():
                    if cond in out_dct[stat][exp][tot]:
                        out_plot[cond] = out_dct[stat][exp][tot][cond]
                        if cond not in args.control:
                            hit_plot[cond] = cmp_dct[stat][exp][cond]

                if out_plot:
                    pth = args.save_path / 'main_base' / f'{stat}-{exp}-{tot}'
                    pth.mkdir(parents=True, exist_ok=True)
                    get_base_plot(args, pth, logging,
                                  out_plot, hit_plot)


def main_cluster(args, cnd_dct, hit_dct,
                 cnd_num, eid=5):
    (args.save_path / 'main_cluster').mkdir(parents=True, exist_ok=True)
    if (args.save_path / 'main_cluster' / 'max_dos.csv').is_file():
        out_dct = None
    else:
        out_dct = run_all_strat(args,
                                args.save_path,
                                cnd_dct, hit_dct)[0]
    get_cluster_plot(args, args.save_path, eid,
                     cnd_num, cnd_dct, out_dct)


def main(args):
    data, plot_dct, cond_dct, hit_dct = get_data(args)

    if args.task in ('stat', 'demo'):
        # prepare the list of models and average images
        model, avgim = [], []
        for ckpt in args.ckpt_path:
            model.append(get_model(str(ckpt), args.decoder, args.encoder))
            avgim.append(get_avgim(model[-1], args.decoder))

        # prepare the keys of the dict storing stats for each iter
        keynm = {'stat': ['mean', 'scm'],
                 'cod': get_codnm(args, len(avgim)),
                 'met': ['psnr', 'ssim']}
        # Statistics: scm, eigenvalue, etc.
        if args.task == 'stat':
            main_stat(args, data, cond_dct,
                      model, avgim, keynm)
        # Demo: images/videos for paper presentation
        else:
            main_demo(args, data, plot_dct,
                      model, avgim)
    # Reconstruction error: psnr and ssim
    elif args.task == 'error':
        main_error(args, cond_dct)
    # Quantification: numerical score measuring heterogeneity
    elif args.task == 'quant':
        main_quant(args, plot_dct, cond_dct, hit_dct)
    # Visualization: pca and phenotypic transition
    elif args.task == 'visual':
        assert args.data_splt == args.task
        model = [get_model(str(ckpt), args.decoder, args.encoder)
                 for ckpt in args.ckpt_path]
        main_visual(args, plot_dct, model)
    # Baseline comparison: Cuccarese et.al. 2020
    elif args.task == 'baseline':
        assert 'rxrx19' in args.data_name
        main_base(args, cond_dct, hit_dct)
    elif args.task == 'cluster':
        assert 'rxrx19' in args.data_name
        # exclude mock, irradiated or healthy
        cond_num = 52 if args.data_name == 'rxrx19a' else 51
        main_cluster(args, cond_dct, hit_dct, cond_num)


if __name__ == '__main__':
    args = parse_args()
    setup_seed(args.seed)
    setup_logg(args)
    main(args)
