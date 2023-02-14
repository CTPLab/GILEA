import copy
import math
import pickle
import random
import torch
import warnings
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.linalg import svdvals, svd, eigvalsh
from sklearn.cluster import KMeans
# from spherecluster import SphericalKMeans

from criteria.psnr import PSNR
from criteria.ms_ssim import MS_SSIM
from style3.utils import common
from style3.utils.data_utils import linspace


def to_cpu(input):
    return input.detach().cpu().numpy()


def to_gpu(input):
    return input.cuda()


def prep_dose(dct, exp=('1', '2')):
    dose = set()
    for e in exp:
        if dct[e] is not None:
            dose.update(list(dct[e].keys()))
    dose = list(dose)
    dose.sort(key=lambda x: float(x))
    return dose


def save_npy(stat_file,
             stat_data):
    if Path(stat_file).exists():
        warnings.warn(f'{stat_file} exist.')
    with open(stat_file, 'wb') as f:
        np.save(f, stat_data)
        print(f'save {stat_file}')


def save_pkl(stat_file,
             stat_data):
    if Path(stat_file).exists():
        warnings.warn(f'{stat_file} exist.')
    with open(stat_file, 'wb') as f:
        pickle.dump(stat_data, f)
        print(f'save {stat_file}')


def add_bbx(input, bbx_clr=(255, 255, 255), bbx_len=2):
    assert len(input.shape) in (3, 4)
    for i in range(3):
        for j in range(4):
            slc = [[None, None], [None, None]]
            # if j == 0, then slc = [[None, bbx_len], [None, None]]
            slc[j // 2][(j + 1) % 2] = (-1) ** j * bbx_len
            # single image, assume the color chn is the last dim
            if len(input.shape) == 3:
                assert input.shape[2] == 3
                slicing = (slice(*slc[0]),) + (slice(*slc[1]),) + (i,)
            # multiple image as a batch, assume the color chn is the 2nd dim
            else:
                assert input.shape[1] == 3
                slicing = (slice(None),) + (i,) + \
                    (slice(*slc[0]),) + (slice(*slc[1]),)
            input[slicing] = bbx_clr[i]
        # input[:bbx_len, :, i] = bbx_clr[i]
        # input[-bbx_len:, :, i] = bbx_clr[i]
        # input[:, :bbx_len, i] = bbx_clr[i]
        # input[:, -bbx_len:, i] = bbx_clr[i]
    return input


def add_contrast(input, axis, weight=1.5):
    assert len(input.shape) in (3, 4)
    assert input.shape[axis] in (1, 3, 6)
    assert 0 <= axis < len(input.shape)

    if input.shape[axis] == 6:
        for ch in range(6):
            slicing = (slice(None),) * axis + (ch,)
            add_wei = 0
            if ch == 2:  # Actin
                add_wei = 0.5
            elif ch == 4:  # Mitochondria
                add_wei = 1.5
            input[slicing] *= (weight + add_wei)
    input[input < 0] = 0
    input[input > 1] = 1
    return input


def get_rxrx19_img(img, axis=1):
    if img.shape[axis] == 6:
        out = add_contrast(img, axis)
        # always append 345 channels along rows
        out = torch.cat([out[:, :3].clone(),
                         out[:, 3:].clone()], axis=-2)
    elif img.shape[axis] == 5:
        mito = torch.zeros([img.shape[0], 1,
                            img.shape[2], img.shape[3]])
        out = torch.cat([img, mito.to(img)], axis=1)
        # always append 345 channels along rows
        out = torch.cat([out[:, :3].clone(),
                         out[:, 3:].clone()], axis=-2)
    return out


def get_bat_img(img, bat):
    # img can be the tuple of image and img_path
    # for the case of rxrx19
    if isinstance(img, (list, tuple)):
        img = img[0]

    # the case of rxrx19
    if len(img.shape) == 5:
        assert img.shape[0] == 1
        # it is possible that only one cell collected from image
        # specifiy the dim = 0 to avoid squeeze the true batch dim
        img = img.squeeze(0)
        num = math.ceil(img.shape[0] / bat)
    else:
        num = 1
    return img, num


def get_cod_dct(pth, log, dat_sub, cod_nam, eig_nam,
                eig_plt=False,
                eig_lst=[1, 2, 3, 4, 5, 10]):
    cod_dct = dict()
    hit_lst = [[[] for _ in eig_lst], [[] for _ in eig_lst]]
    cmpval, cmpvec, cmpavg = [], [], []
    for sid, sub in enumerate(dat_sub):
        hit = [[0 for _ in eig_lst], [0 for _ in eig_lst]]
        cod_dct[sub] = {'val': [], 'vec': [], 'avg': []}
        for cid, cod in enumerate(cod_nam):
            pthval = pth / sub / f'{cod}_{eig_nam}val.npy'
            eigval = np.load(str(pthval))

            pthvec = pth / sub / f'{cod}_{eig_nam}vec.npy'
            eigvec = np.load(str(pthvec))

            pthavg = pth / sub / f'{cod}_mean.npy'
            eigavg = np.load(str(pthavg))

            if eig_plt:
                cod_dct[sub]['val'].append(eigval)
                cod_dct[sub]['vec'].append(eigvec)
                cod_dct[sub]['avg'].append(eigavg)
            if sid == 0:
                cmpval.append(eigval)
                cmpvec.append(eigvec)
                cmpavg.append(eigavg)

            msg = f'{sub}_{cod}: '
            log.info(msg + '{}, {}'.format(eigval[:5], eigvec.shape))
            # avgcmp = cod_dct[dat_sub[0]]['avg'][cid]

            # print(eigval.shape, eigcmp.shape)
            # for eid, est in enumerate(eig_lst):
            #     val0 = np.expand_dims(eigval[:est], -1)
            #     val1 = np.expand_dims(cmpval[cid][:est], -1)
            #     diff = (val0 * eigvec[:, :est].T -
            #             val1 * cmpvec[cid][:, :est].T) / val1
            #     diag = np.diag(diff @ diff.T)
            #     print(diag)
            #     # print(val0.shape, val1.shape, diff.shape, diag.shape)
            #     hit[0][eid] += diag.sum()
            for eid, est in enumerate(eig_lst):
                diff = (np.sqrt(eigval[:est]) -
                        np.sqrt(cmpval[cid][:est])) / np.sqrt(cmpval[cid][:est])
                hit[0][eid] += diff.dot(diff)
                diff = np.diag(cmpvec[cid][:, :est].T @ eigvec[:, :est])
                hit[1][eid] -= np.abs(diff).sum()
                # if 'cov' == cod_nam:
                #     diff_avg = (avgval[:est] - avgcmp[:est]) / avgcmp[:est]
                #     hit[eid] += diff_avg.dot(diff_avg)
        for eid, est in enumerate(eig_lst):
            hit_lst[0][eid].append(hit[0][eid])
            hit_lst[1][eid].append(hit[1][eid])
    for eid, est in enumerate(eig_lst):
        for hid, hnm in enumerate(('val', 'vec')):
            hit_rnk = np.argsort(hit_lst[hid][eid])
            hit_res = list(zip(dat_sub, hit_lst[hid][eid]))
            msg = f'hit_score {hnm} ({est}): '
            for hit in hit_rnk:
                msg += f'{hit_res[hit]} '
            log.info(msg + '\n')
        log.info('\n')
    return cod_dct


def get_eig_dct(pth, log, dat_sub, cod_nam, eig_nam):
    cod_dct = dict()
    hit_lst = list()
    for sid, sub in enumerate(dat_sub):
        hit = 0
        cod_dct[sub] = {'val': list(), 'vec': list()}
        for cid, cod in enumerate(cod_nam):
            pthscm = pth / sub / f'{cod}_{eig_nam}.npy'
            scm = np.load(str(pthscm))
            eigval = eigvalsh(scm)
            eigval = eigval[::-1]
            cod_dct[sub]['val'].append(eigval)

            msg = f'{sub}_{cod}: '
            log.info(msg + '{}_{}'.format(eigval[:5], eigval[-5:]))
            eigcmp = cod_dct[dat_sub[0]]['val'][cid]
            # print(eigval.shape, eigcmp.shape)
            diff = (eigval[:5] - eigcmp[:5]) / eigcmp[:5]
            hit += diff.dot(diff)
        hit_lst.append(hit)
    hit_rnk = np.argsort(hit_lst)
    hit_res = list(zip(dat_sub, hit_lst))
    msg = 'hit_score: '
    for hit in hit_rnk:
        msg += f'{hit_res[hit]} '
    log.info(msg)
    return cod_dct


def get_cluster_dct(arg,
                    dat_sub,
                    cod_nam,
                    cod_dct,
                    pca_cls=3,
                    pca_num=16,
                    pca_dim=100,
                    pca_max=50000):
    pca_dct, pca_idx = dict(), dict()
    lay_num, lay_dim = cfg_cod_shape(arg)
    for sub in dat_sub:
        pca_dct[sub] = [[] for _ in range(pca_cls)]
        pca_idx[sub] = [[] for _ in range(pca_cls)]
        for cid, cod in enumerate(cod_nam):
            # only use 'cod0' to compute cluster, may not
            # lead to meaningful clustering for layer-wise cases
            eig_vec = (cod_dct[sub]['vec'][cid].T)[:pca_dim]
            pth_cod = arg.save_path / sub / f'{cod}_kid'
            mat_cod = np.load(str(pth_cod) + '.npy')
            if mat_cod.shape[1] > pca_max:
                mat_idx = list(range(mat_cod.shape[1]))
                random.shuffle(mat_idx)
                mat_cod = mat_cod[:, mat_idx[:pca_max]]

            pca = eig_vec @ mat_cod
            kmeans = KMeans(n_clusters=pca_cls, random_state=0).fit(pca.T)
            kcents = kmeans.cluster_centers_
            # sort the centriods based on the largest spike
            kcents = kcents[kcents[:, 0].argsort()]
            print(pca.T.shape, mat_cod.shape)
            for kid, kct in enumerate(kcents):
                if cid == 0:
                    # compute l2 distance to centroid
                    dist = ((pca.T-kct)**2).sum(axis=1)
                    pca_idx[sub][kid].append(dist.argsort()[:pca_num])
                # for each list of latent representations
                # extract pca_num ones that are closest to centroid
                vis_ids = pca_idx[sub][kid][cid]
                vis_cod = mat_cod[:, vis_ids].T
                vis_cod = vis_cod.reshape((pca_num, lay_num, lay_dim))
                vis_cod = torch.from_numpy(vis_cod).cuda().float()
                pca_dct[sub][kid].append(vis_cod)
                print(sub, kct, pca_dct[sub][kid][0].shape)
    return pca_dct

    # if sub not in ('storm-severe', 'healthy'):
    #     with open(str(pth_kid) + '.pickle', 'rb') as f:
    #         pca_dct[sub].append(pickle.load(f))
    #     pca_dct[sub].append(kmeans.labels_)
    #     np.save(str(pth_kid) + '_cluster.npy', klabel)

    # pca_vis.append(pca)
    # pca_clr.extend([dat_clr[sub]] * pca.shape[-1])
    # pca_cnt = np.unique(klabel, return_counts=True)[1]
    # print(sub, cod, pca_cnt / pca_cnt.sum())
    # print(sub, cod, pca.shape, dat_clr[sub], kmeans.cluster_centers_, kcents, klabel.shape)
    # print(np.unique(kmeans.labels_, return_counts=True))
    # print(np.unique(klabel, return_counts=True))
    # pca_vis = np.concatenate(pca_vis, axis=-1)
    # print(pca_vis.shape)
    # plt.scatter(pca_vis[0], pca_vis[1], c=pca_clr, s=10)
    # plt.show()
    # return pca_dct


def get_man_msk(codes, lay_num, is_layer):
    if is_layer:
        assert (len(codes) // lay_num) * lay_num == len(codes)
        # if the codes len > lay_num, it means not is_total.
        # Then each channel is independent, so we could use np.tile
        # and manipulate eigenvalue for each channel together
        out_msk = torch.tile(torch.eye(lay_num),
                             [len(codes) // lay_num]).cuda()
    else:
        out_msk = [torch.ones(len(codes)).cuda()]
    return out_msk


def run_one_enc(arg, model, avgim, input):
    input = input.clone()
    input = input.float() / 127.5 - 1
    # if multiple avarage images, then the loaded models
    # are trained on images with each individual channel
    if len(avgim) != 1:
        input = torch.split(input, 1, dim=1)
        assert len(avgim) == len(input)
    else:
        input = [input]

    codes = []
    for mod, avg, inp in zip(model, avgim, input):
        avg = avg.unsqueeze(0)
        avg = avg.repeat(inp.shape[0], 1, 1, 1)
        # concatenate the average image to input
        avinp = torch.cat([inp, avg], dim=1)
        with torch.inference_mode():
            cod = mod.encoder(avinp)
            if not arg.stat_res:
                avlat = mod.latent_avg.repeat(avinp.shape[0], 1, 1)
                cod += avlat.to(avinp)
            codes.append(cod)
    return codes


def run_one_dec(arg, model, codes):
    outs = []
    for mod, cod in zip(model, codes):
        # if lay_dup != 1:
        #     cod = cod.repeat(1, 1, lay_dup)

        if arg.stat_res:
            avlat = mod.latent_avg.repeat(cod.shape[0], 1, 1)
            cod = avlat.to(cod) + cod
        if arg.decoder == 'style2':
            # during recon the model will add small random noise
            # this cause minor variance of the reconstruction
            with torch.inference_mode():
                out = mod.decoder([cod],
                                  input_is_latent=True)[0]

        elif arg.decoder == 'style3':
            id_trans = common.get_identity_transform()
            id_trans = torch.from_numpy(id_trans).unsqueeze(0)
            id_trans = id_trans.repeat(cod.shape[0], 1, 1).cuda().float()
            mod.decoder.synthesis.input.transform = id_trans
            with torch.inference_mode():
                out = mod.decoder.synthesis(cod,
                                            noise_mode='const',
                                            force_fp32=True)

        outs.append(out)
    outs = torch.cat(outs, dim=1)
    outs = (outs + 1) / 2
    outs[outs < 0] = 0
    outs[outs > 1] = 1
    return outs


def cfg_cod_shape(arg):
    num, dim = arg.lay_num, arg.lay_dim
    if arg.is_merge and not arg.is_total:
        dim *= arg.img_chn
    if arg.is_layer:
        num = 1
    return num, dim


def cfg_cod_forward(arg, codes):
    if arg.is_merge and not arg.is_total:
        assert len(codes) > 1
        codes = [torch.cat(codes, dim=2)]

    if arg.is_layer:
        # torch.split outputs a tuple
        codes = [list(torch.split(cod, 1, dim=1)) for cod in codes]
        codes = sum(codes, [])
    return codes


def cfg_cod_backward(arg, codes):
    if arg.is_layer:
        codes = [torch.cat(codes, dim=1)]
        assert codes[0].shape[1] >= arg.lay_num
        if codes[0].shape[1] > arg.lay_num:
            codes = list(torch.split(codes[0], arg.lay_num, dim=1))

    if arg.is_merge and not arg.is_total:
        codes = list(torch.split(codes, arg.lay_dim, dim=2))

    return codes


def run_one_eig(cmp0, cmp1,
                is_cov, cod_dct,
                man_val,
                man_vec,
                man_cod,
                man_pow,
                man_base,
                man_axis):
    clen = len(man_cod)
    assert clen == len(cod_dct[cmp0]['val'])
    assert clen == len(cod_dct[cmp0]['vec'])
    assert clen == len(cod_dct[cmp0]['avg'])
    assert clen == len(man_pow)

    if any(mpow != 0 for mpow in man_pow):
        curval, nxtval = [], []
        curvec, nxtvec = [], []
        curavg, nxtavg = [], []
        weight = []
        for cid in range(clen):
            cval = cod_dct[cmp0]['val'][cid].astype(np.float32)
            curval.append(cval)
            nval = cod_dct[cmp1]['val'][cid].astype(np.float32)
            nxtval.append(nval)

            # Eigenvalue manipulation (Eq. 6 in the paper)
            pow = man_pow[cid] * \
                np.sign(nxtval[cid][man_axis] - curval[cid][man_axis])
            wei = np.power(man_base, pow)

            if isinstance(man_axis, int):
                wei = np.atleast_1d(wei)
            else:
                wei = np.expand_dims(wei, axis=-1)
            weight.append(wei)

            print(man_pow, wei, wei.shape)
            print(man_axis, man_val, man_vec,
                  curval[cid][man_axis],
                  nxtval[cid][man_axis])

            cvec = cod_dct[cmp0]['vec'][cid].astype(np.float32)
            curvec.append(cvec)
            nvec = cod_dct[cmp1]['vec'][cid].astype(np.float32)
            nxtvec.append(nvec)

            cavg = cod_dct[cmp0]['avg'][cid].astype(np.float32)
            curavg.append(cavg)
            navg = cod_dct[cmp1]['avg'][cid].astype(np.float32)
            nxtavg.append(navg)

        out_cod = []
        for cid in range(clen):
            codes = man_cod[cid].clone()
            n, c, d = codes.shape
            codes = codes.reshape(n, -1)

            eigvec = nxtvec[cid] if man_vec else curvec[cid]
            eigvec = torch.from_numpy(eigvec).to(codes)
            eigavg = curavg[cid]
            eigavg = torch.from_numpy(eigavg).to(codes)
            # codes[0,] = eigavg
            if is_cov:
                print('avg', eigavg.shape, codes.shape)
                codes -= eigavg
            codes = eigvec.T @ (codes.T)

            if man_val:
                wei = torch.from_numpy(weight[cid]).to(codes)
                codes[man_axis] = wei * codes[man_axis]
            codes = (eigvec @ codes).T
            if is_cov:
                codes += eigavg
            codes = codes.reshape(n, c, d)
            out_cod.append(codes)
    else:
        out_cod = man_cod
    return out_cod


def run_one_eig_new(arg, base, wei,
                    cod, vec):
    if len(cod.shape) == 3:
        c, n = cod.shape[:2]
        l, d = arg.lay_num, cod.shape[2] // arg.lay_num
    elif len(cod.shape) == 4:
        c, n, l, d = cod.shape
        cod = cod.reshape(c, n, l*d)
    if (wei != 0).any():
        cod = cod @ vec
        wei = torch.pow(base, wei)
        cod = wei * cod
        cod = cod @ vec.transpose(-1, -2)
    cod = cod.reshape(c, n, l, d)
    if not arg.is_total:
        # 128 -> 512
        cod = cod.repeat(1, 1, 1, 4)
    cod = [cod[c] for c in range(len(cod))]
    return cod
