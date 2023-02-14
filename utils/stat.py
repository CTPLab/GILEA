import torch
import numpy as np

from criteria.psnr import PSNR
from criteria.ms_ssim import MS_SSIM

from torch.linalg import svdvals, svd, eigh, eigvalsh


def prep_met(met_key=('psnr', 'ssim'), met_chn=3):
    # collect single channel and total channels met
    met_fuc, met_all = dict(), dict()
    for met in met_key:
        met = met.lower()
        if met == 'psnr':
            met_fuc[met] = (PSNR(dim=[1, 2]),  # single channel, only two dims
                            PSNR(dim=[1, 2, 3]))  # total channels, three dims
        elif met == 'ssim':
            # kernel size = 7 for image res 128x128
            met_fuc[met] = (MS_SSIM(1., False, 7, channel=1, unsqueeze=True),
                            MS_SSIM(1., False, 7, channel=met_chn))
        else:
            raise NameError(f'{met} is not a valid metric.')

        # here we should use list comprehension to avoid
        # list memory sharing issue
        met_all[met] = [[] for _ in range(met_chn + 1)]
    return met_fuc, met_all


def run_one_met(input,
                output,
                met_fuc,
                met_all):
    # print(input.shape, output.shape)
    for mkey, mfuc in met_fuc.items():
        mchn = len(met_all[mkey])
        for chn in range(mchn):
            cn = list(range(chn)) if chn == mchn - 1 else chn
            id = 1 if chn == mchn - 1 else 0
            metric = mfuc[id](input[:, cn],
                              output[:, cn]).detach()
            # print(mkey, cn, id,  metric.shape)
            metric = metric.cpu().numpy().tolist()
            met_all[mkey][chn].extend(metric)


def post_met(pth, met_all):
    for mkey, mlst in met_all.items():
        for mchn in range(len(mlst)):
            stat_pth = str(pth).replace('.pt', f'{mkey}_{mchn}.pt')
            torch.save(torch.tensor(mlst[mchn]), stat_pth)


def prep_stat(cod_num):
    stat_buf = [torch.FloatTensor().cuda() for _ in range(cod_num)]
    stat_all = {'mean': [0 for _ in range(cod_num)],
                'scm': [0 for _ in range(cod_num)]}
    return stat_buf, stat_all


def run_one_stat(codes,
                 stat_all,
                 stat_buf,
                 save_low=True,
                 save_buf=True):
    for cid, cod in enumerate(codes):
        if save_low:
            for i in range(1, 4):
                assert (cod[:, :, :128] ==
                        cod[:, :, i * 128: (i + 1) * 128]).all()
            cod = cod[:, :, :128].clone()
        cod = cod.contiguous().view(cod.shape[0], -1)
        if save_buf:
            stat_buf[cid] = torch.cat((stat_buf[cid], cod))
        cod = cod.double()
        stat_all['mean'][cid] += cod.sum(dim=0)
        stat_all['scm'][cid] += cod.T @ cod


def run_one_stat1(codes,
                  tsr_low,
                  stat_buf,
                  stat_all,
                  stat_cmp,
                  first=False,
                  debug=False):

    assert len(list(tsr_low.keys())) == len(codes)
    for tid, tsr in enumerate(tsr_low.keys()):
        tsr_val = codes[tid].contiguous().view(codes[tid].shape[0], -1)
        tsr_val = tsr_val.double()

        # if i == 0 and n == 0:
        if first:
            print(tsr, tsr_val.shape, tsr_low[tsr])
            for stat in stat_buf:
                stat_buf[stat][tsr] = list()
                stat_all[stat][tsr] = list() if tsr_low[tsr] else 0
                stat_cmp[stat][tsr] = 0 if tsr_low[tsr] else list()

        for stat in stat_buf:
            stat_cur = batch_stat(tsr_val,
                                  stat)
            stat_buf[stat][tsr] += [stat_cur]
            if debug:
                # works for rxrx19b and ham10k,
                # may not work for other cases
                stat_cmp[stat][tsr] += align_stat(stat_cur,
                                                  stat,
                                                  not tsr_low[tsr])


def batch_stat(stat, name):
    # assume the input is (batch, spatial_dim)
    assert name in ('mean', 'scm')
    if name == 'mean':
        stat = torch.sum(stat, dim=0)
    else:
        stat = stat.T
    return stat


def align_stat(stat, name, lrank):
    if lrank:
        return [stat]
    else:
        if name != 'mean':
            return stat @ stat.T
        return stat


def join_stat(stat, name, lrank):
    if lrank:
        stat = sum(stat) if name == 'mean' else \
            torch.cat(stat, dim=1)
    return stat


def calc_stat(stat,
              topk,
              basis=False):
    if basis:
        eigvec, eigval = svd(stat, full_matrices=False)[:2]
    else:
        eigvec = None
        if topk != -1:
            eigval = torch.lobpcg(stat, k=topk)[0]
        else:
            eigval = svdvals(stat)
    return eigval, eigvec


def buff_stat(stat_buf,
              save_pth):
    feat_out = torch.stack(stat_buf)
    stat_pth = str(save_pth).replace('.pt', 'kid.pt')
    torch.save(feat_out, stat_pth)


def comp_stat(cmp0,
              cmp1,
              stat,
              tsr):
    # this function not working anymore
    msg = '({})'.format(tsr)
    if stat == 'mean':
        cmp_err = np.max(np.abs(cmp0[0] - cmp1[0]).flatten())
        msg += ' mean: {}'.format(cmp_err)
    else:
        for cid, (c0, c1) in enumerate(zip(cmp0, cmp1)):
            if cid == 0:
                if c0.shape[0] != c0.shape[1]:
                    assert c1.shape[0] == c1.shape[1]
                    c0 = c0 @ c0.T
                else:
                    assert c1.shape[0] != c1.shape[1]
                    c1 = c1 @ c1.T
                cmp_err = np.max(np.abs(c0 - c1).flatten())
                msg += ' {}: {}'.format(stat, cmp_err)
            elif c0 is not None:
                print(c0.shape, c1.shape)
                if cid == 1:
                    assert len(c0.shape) == len(c1.shape) == 1
                    min_len = min(c0.shape[0], c1.shape[0])
                    c0, c1 = c0[:min_len], c1[:min_len]
                elif cid == 2:
                    assert len(c0.shape) == len(c1.shape) == 2
                    min_len = min(c0.shape[1], c1.shape[1])
                    c0, c1 = c0[:, :min_len], c1[:, :min_len]
                cmp_err = np.max(np.abs(c0 - c1).flatten())
                msg += ' {}: {}'.format('eigval' if cid == 1 else 'eigvec',
                                        cmp_err)
    print(msg)


def post_stat(path, size,
              stat, topk, strat=False):

    for key, val in stat.items():
        # normalize input to mean or sample covariance with '/ size'
        out = torch.stack(val) / size
        pth = str(path).replace('.pt', f'{key}.pt')
        print(key, out.shape)

        if key == 'mean':
            avg = out
            torch.save(avg, pth)
        elif key == 'scm':
            assert out.shape[-1] == out.shape[-2]
            scm = calc_stat(out, topk,
                            not strat)
            torch.save(scm, pth)
            print('scm', scm[0].shape, scm[0][:, :5])
            cov = calc_stat(out - torch.einsum('bi,bj->bij', avg, avg),
                            topk,
                            not strat)
            torch.save(cov, pth.replace(key, 'cov'))
            print('cov', cov[0].shape, cov[0][:, :5])
