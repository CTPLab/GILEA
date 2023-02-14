import sys
import torch
import imageio
import argparse
import pandas as pd

from umap import UMAP
from pathlib import Path
from argparse import Namespace
from sklearn.manifold import TSNE

from cfgs import Config
from utils.demo import get_raw_img, get_vid_img
from utils.common import get_bat_img, run_one_enc, run_one_dec, cfg_cod_forward, to_gpu, prep_dose
from utils.stat import prep_met, prep_stat, run_one_met, run_one_stat, buff_stat, post_stat, post_met
from utils.plot import calc_deig, set_res_plot, app_dose, acc_deig, set_res_plot, get_pca_res, get_pca_plot, get_man_img
sys.path.append('.')


def run_all_demo(arg, ref, cnd,
                 stat, colr, fpth,
                 dload, model, avgim):
    spth = arg.save_path / 'main_demo'
    spth.mkdir(parents=True, exist_ok=True)

    for _, (inp, _, meta) in enumerate(dload):
        if arg.data_name == 'ham10k':
            img, cfg = inp.cuda(), None
        elif 'rxrx19' in arg.data_name:
            assert inp[0].shape[0] == 1 and len(inp[0].shape) == 5
            img = inp[0].squeeze().cuda()

            cfg = Config().rxrx19[arg.data_cell]
            cfg = Namespace(**cfg)
            rdir = str(arg.data_path).replace('_cell', '')
            rpth = Path(rdir) / 'images' / meta[0][0] / \
                f'Plate{meta[1][0]}' / f'{meta[2][0]}_s{meta[3][0]}_w1.png'
            raw_img, chn_img = get_raw_img(cfg, ref, cnd,
                                           rpth, spth,
                                           colr, fpth)
            cfg.pos, cfg.raw_img, cfg.chn_img = inp[1], raw_img, chn_img

        # prepare the video writer
        writers = []
        for chn in ('R', 'G', 'B', 'all'):
            vid_file = f'{ref}_{cnd}_eig{arg.demo_axis}_chn{chn}_{arg.seed}.mp4'
            writers.append(imageio.get_writer(str(spth / vid_file),
                                              fps=24))

        dim, axs = 64, arg.demo_axis
        vec = stat[ref]['vec'] if arg.demo_vec else stat[cnd]['vec']
        wei = torch.zeros_like(stat[ref]['val']).unsqueeze(1)
        wei[:, :, axs] = 1
        get_vid_img(arg, img,
                    wei, vec, dim,
                    model, avgim, writers,
                    cfg)

        for wid, wrt in enumerate(writers):
            wrt.close()
        break


def run_strat_sum(size0, size1,
                  stat0, stat1,
                  met0, met1,
                  topk,
                  path=None,
                  strat=True):
    size = size0 + size1

    # since we always set lrank = False
    # we can add x1 @ x1.T and x2 @ x2.T directly
    stat = dict()
    for key, cod_lst in stat0.items():
        stat[key] = [None for _ in range(len(cod_lst))]
        for cid in range(len(cod_lst)):
            stat[key][cid] = stat0[key][cid] + stat1[key][cid]

    met = dict()
    for key, lst in met0.items():
        met[key] = list()
        for mid in range(len(lst)):
            met[key].append(met0[key][mid] + met1[key][mid])

    if path is not None:
        post_stat(path, size,
                  stat, topk, strat=strat)
        post_met(path, met)
    return size, stat, met


def run_all_stat(arg,
                 pth,
                 total,
                 dload,
                 model,
                 avgim,
                 keynm,
                 bufer=False,
                 strat=False):
    print(f'the pre-calc cell num {total}')

    # prepare the statistics
    stat_buf, stat_all = prep_stat(len(keynm['cod']))

    # prepare the image metrics
    met_fuc, met_all = prep_met(met_key=keynm['met'],
                                met_chn=arg.img_chn)

    # run the data loader to collect all the stats/metrics
    tot = 0
    for _, (img, _, meta) in enumerate(dload):
        img, num = get_bat_img(img, arg.n_eval)
        img = to_gpu(img)

        for n in range(num):
            input = img[n * arg.n_eval: (n + 1) * arg.n_eval]
            if len(input.shape) == 3:
                input = input.unsqueeze(0)

            codes = run_one_enc(arg, model, avgim, input)
            if arg.stat_dec:
                output = run_one_dec(arg, model, codes)
            codes = cfg_cod_forward(arg, codes)
            # calc one batch of codes stats
            run_one_stat(codes, stat_all, stat_buf, not arg.is_total, bufer)
            # calc one batch image metrics
            if arg.stat_dec:
                run_one_met(input.float().div(255.),
                            output,
                            met_fuc,
                            met_all)

            tot += input.shape[0]
        # buffer for rxrx19a/b
        if 'rxrx19' in str(arg.data_name) and bufer and tot >= arg.img_buf:
            buff_stat(stat_buf, pth)
            bufer = False

    assert total == tot, \
        f'The precomp data amount {total} != train data amount {tot}'
    if bufer:
        buff_stat(stat_buf, pth)

    post_stat(pth, tot,
              stat_all,
              arg.stat_top,
              strat=strat)

    post_met(pth, met_all)
    return stat_all, met_all


def run_all_strat(args, path, cond_dct, hit_dct):
    ref = 'Mock' if args.data_name == 'rxrx19a' else 'healthy'
    out_dct, cmp_dct, ref_dct = {}, {}, {}
    for stat in ('scm', 'cov'):
        out_dct[stat], cmp_dct[stat], ref_dct[stat] = {}, {}, {}
        for exp in ('1', '2', 'all'):
            out_dct[stat][exp] = {'max_dos': {}, 'all_dos': {}, 'one_dos': {}}
            cmp_dct[stat][exp] = {}
            if args.data_cell != 'HUVEC' or exp == '1':
                key = '' if exp == 'all' else f'-{exp}'
                pth = path / ref / f'{ref}{key}_{stat}.pt'
                ref_dct[stat][exp] = torch.load(pth)[0]
            elif exp == 'all':
                # for the huvec case and exp 'all'
                pth = path / ref / f'{ref}-1_{stat}.pt'
                ref_dct[stat][exp] = torch.load(pth)[0]
    print(ref_dct)

    for cond, cval in cond_dct.items():
        stat_pth = path / cond
        # get all the doses for 1 or 2 experiments
        dose = prep_dose(cval, ('1', '2'))
        print(cond, dose)

        for dos in dose:
            for exp in ('1', '2'):
                if cval[exp] is not None:
                    for stat in ('scm', 'cov'):
                        ref = ref_dct[stat][exp]
                        ref_all = ref_dct[stat]['all']
                        key = f'{cond}-{exp}_{stat}.pt'
                        key_dos = f'{cond}-{exp}-{dos}_{stat}.pt'

                        if cond in args.control:
                            assert dos == dose[-1]
                            # calc deig of controlled cond
                            out = calc_deig(ref, stat_pth / key)
                            out_all = calc_deig(ref_all, stat_pth / key)
                            set_res_plot(stat, exp, cond,
                                         out_dct, out, out_all)
                        else:
                            # calc deig of drugs
                            out = calc_deig(ref, stat_pth / key_dos)
                            acc_deig(stat, exp, cond,
                                     out_dct, out)

                            app_dose(stat, exp, cond,
                                     out_dct, dos, out)

                            out_all = calc_deig(ref_all, stat_pth / key_dos)
                            acc_deig(stat, 'all', cond,
                                     out_dct, out_all)

                            app_dose(stat, 'all', cond,
                                     out_dct, dos, out_all)

                            if dos == dose[-1]:
                                out = calc_deig(ref, stat_pth / key)
                                out_dct[stat][exp]['all_dos'][cond] = out

                                out = calc_deig(ref_all, stat_pth / key)
                                out_dct[stat]['all']['all_dos'][cond] = out

                                # assign the existing hit scores
                                hit = hit_dct[cond][int(exp) - 1]
                                cmp_dct[stat][exp][cond] = hit
                                cmp_dct[stat]['all'][cond] = hit

        if cval['1'] is not None and cval['2'] is not None:
            for stat in ('scm', 'cov'):
                ref = ref_dct[stat]['all']
                key = f'{cond}_{stat}.pt'
                out = calc_deig(ref, stat_pth / key)
                out_dct[stat]['all']['all_dos'][cond] = out

                if cond in args.control:
                    out_dct[stat]['all']['max_dos'][cond] = out
                    out_dct[stat]['all']['one_dos'][cond] = out
                else:
                    del out_dct[stat]['all']['max_dos'][cond]
                    out_dct[stat]['all']['one_dos'][cond] = dict()
                    for dos in dose:
                        key_dos = f'{cond}-{dos}_{stat}.pt'
                        out_all = calc_deig(ref, stat_pth / key_dos)
                        acc_deig(stat, 'all', cond,
                                 out_dct, out_all)
                        app_dose(stat, 'all', cond,
                                 out_dct, dos, out_all)
                    hit0 = hit_dct[cond][0]
                    hit1 = hit_dct[cond][1]
                    cmp_dct[stat]['all'][cond] = max(hit0, hit1)
    return out_dct, cmp_dct


def run_all_vis(arg, slc,
                eig_dct, plt_dct,
                model, red_dim='PC'):

    # assume key[0] is reference
    key = list(eig_dct.keys())
    ref_val = eig_dct[key[0]]['val'].unsqueeze(1)
    ref_vec = eig_dct[key[0]]['vec'][:, :, :slc + 1]
    ref_vec = ref_vec.transpose(-1, -2)
    out, grp, chn = {}, {}, len(eig_dct[key[0]]['cod'])
    for eid, eig in eig_dct.items():
        print(slc, eid)
        if red_dim == 'PC':
            pca_sgn = ref_vec @ eig['vec'][:, :, :slc + 1]
            pca_sgn = torch.diagonal(pca_sgn, dim1=-2, dim2=-1).unsqueeze(1)
            print(pca_sgn, pca_sgn.shape)
            pca = eig['cod'] @ eig['vec'][:, :, :slc + 1]
            # pca = pca * pca_sgn
            grp[eid] = pca[:, :, :slc].cpu()
            pca = pca ** 2 / ref_val[:, :, :slc + 1]
            pca[:, :, slc] = torch.sum(pca[:, :, 1:slc], dim=2)
            pca_all = torch.sum(pca, dim=0).unsqueeze(0)
            out[eid] = torch.cat((pca, pca_all)).cpu()
        elif red_dim in ('t-SNE', 'UMAP'):
            if red_dim == 't-SNE':
                fn = TSNE()
                cod = eig['cod'] @ eig['vec'][:, :, :50]
            else:
                fn = UMAP()
                cod = eig['cod']
            cod = cod.cpu().numpy()
            red = [torch.from_numpy(fn.fit_transform(cod[c]))
                   for c in range(chn)]
            out[eid] = torch.stack(red, dim=0)

    if red_dim == 'PC':
        slc += 1
        if 'rxrx19' in arg.data_name:
            chn += 1

        cnd_val = eig_dct[key[-1]]['val'].unsqueeze(1)
        # assume key[-1] is condtion of interest
        key_vec = key[0] if arg.plot_vec else key[-1]
        vec = eig_dct[key_vec]['vec']
        for c in range(chn):
            # only check pc-1,2 and pc-1, 2-5
            for s in (1, slc - 1):
                # if c < chn - 1 and s < slc - 1:
                #     continue

                pth = arg.save_path / 'main_visual' / \
                    f'{key[-1]}_{red_dim}_1_{s + 1}'
                pth.mkdir(parents=True, exist_ok=True)

                wsl, col = [0, s], [f'{red_dim}_1', f'{red_dim}_{s + 1}']
                if s == slc - 1 and slc > 1:
                    wsl = list(range(s))
                    col = [f'{red_dim}_1', f'{red_dim}_2_{s}']

                for eid in eig_dct:
                    if c < chn-1:
                        cod_dct, kct_df = get_pca_res(col,
                                                      eig_dct[eid]['cod'],
                                                      grp[eid][c, :,
                                                               wsl].numpy(),
                                                      ref_val[c, :, wsl].cpu().numpy())
                    else:
                        cod_dct, kct_df = get_pca_res(col,
                                                      eig_dct[eid]['cod'],
                                                      grp[eid][:, :,
                                                               wsl].numpy(),
                                                      ref_val[:, :, wsl].cpu().numpy())

                    kct_df['color'] = plt_dct[eid][0]
                    for sl in ([0], [s], wsl):
                        print(key[-1], c, sl)
                        wei = torch.zeros_like(ref_val)
                        if c < chn-1:
                            wei[c, :, sl] = 1
                        else:
                            wei[:, :, sl] = 1
                        get_man_img(arg, pth, c, sl,
                                    plt_dct[eid][-1], wei, eig_dct[eid]['vec'],
                                    cod_dct, model, eid)

                get_pca_plot(pth, c, col, (0, s),
                             out, plt_dct,
                             None, red_dim)

    elif red_dim in ('t-SNE', 'UMAP'):
        pth = arg.save_path / 'main_visual' / f'{key[-1]}_{red_dim}'
        pth.mkdir(parents=True, exist_ok=True)
        for c in range(chn):
            get_pca_plot(pth, c, (0, 1),
                         out, plt_dct,
                         None, red_dim)


def dataset_stat(csv_dir):
    df = pd.read_csv(str(csv_dir / 'metadata_HRCE.csv'))
    # df0 = df[df.cell_type == 'VERO']
    # df0.to_csv(str(csv_dir / 'metadata_VERO.csv'), index=False)
    # df1 = df[df.cell_type == 'HRCE']
    # df1.to_csv(str(csv_dir / 'metadata_HRCE.csv'), index=False)
    if 'ham10k' in str(csv_dir):
        for col, col_val in df.iteritems():
            # ignore lesion_id and image_id
            if '_id' not in col:
                print(pd.value_counts(col_val).sort_index())
                print()
    elif 'rxrx19' in str(csv_dir):
        # # df = df[df.cell_type == 'HRCE']
        # # df = df[df.disease_condition.isna()]
        # df = df[df.cell_type == 'VERO']
        # df = df[df.disease_condition == 'Active SARS-CoV-2']
        for col, col_val in df.iteritems():
            # print(col)
            # ignore site_id, well_id, well, SMILE
            # if '_id' not in col and \
            #    'well' not in col and \
            #    'SMILES' not in col and 'treatment' != col:
            # for key, val in pd_dict.items():
            #     print(key, val)
            # if 'treatment' == col:
            # if 'cell_type' == col:
            # if 'disease_condition' == col:
            if col in ('cell_type', 'experiment', 'disease_condition', 'treatment'):
                pd_dict = pd.value_counts(col_val, dropna=False, sort=True)
                keys = list(pd_dict.keys())
                # print(pd_dict, len(keys))
                for kid, key in enumerate(keys[:500]):
                    print(key, pd_dict[key])
                # for key, val in pd_dict.items():
                #     print(key, val)


def main():
    parser = argparse.ArgumentParser(
        description='Collect data stats')

    parser.add_argument(
        '--csv_dir',
        type=Path,
        default=Path('Data/non_IID/preproc/rxrx19a/'),
        help='dir of the metadata csv')

    args = parser.parse_args()
    dataset_stat(args.csv_dir)


if __name__ == '__main__':
    main()
