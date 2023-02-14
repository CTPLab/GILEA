
import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.utils as tv_utils

from pathlib import Path
from sklearn.cluster import KMeans
from adjustText import adjust_text
from matplotlib.ticker import StrMethodFormatter
from scipy.linalg import svdvals, eigvals
from scipy.stats import ks_2samp, ttest_ind, mannwhitneyu, levene, f

from utils.common import add_contrast, add_bbx, prep_dose, run_one_eig_new, run_one_dec
font = {'family': 'normal',
        'weight': 'bold',
        'size': 20}
plt.rc('font', **font)


def lowess_with_confidence_bounds(
    x, y, eval_x=None, N=200, conf_interval=0.95, lowess_kw=None
):
    """
    Perform Lowess regression and determine a confidence interval by bootstrap resampling
    """
    # Lowess smoothing
    smoothed = sm.nonparametric.lowess(
        exog=x, endog=y, xvals=eval_x, **lowess_kw)

    # Perform bootstrap resamplings of the data
    # and  evaluate the smoothing at a fixed set of points
    smoothed_values = np.empty((N, len(eval_x)))
    for i in range(N):
        sample = np.random.choice(len(x), len(x), replace=True)
        sampled_x = x[sample]
        sampled_y = y[sample]

        smoothed_values[i] = sm.nonparametric.lowess(
            exog=sampled_x, endog=sampled_y, xvals=eval_x, **lowess_kw
        )

    # Get the confidence interval
    sorted_values = np.sort(smoothed_values, axis=0)
    bound = int(N * (1 - conf_interval) / 2)
    bottom = sorted_values[bound - 1]
    top = sorted_values[-bound]

    return smoothed, bottom, top


def gen_msg(dat, chn, met, sub):
    msg = '({} {} {}): mean {:.3f}, std {:.3f}'.format(
        chn, met, sub,
        torch.mean(dat),
        torch.std(dat))
    return msg


def calc_deig(ref, pth,
              eig_lst=[1, 2, 3, 4, 5, 10]):
    cnd = torch.load(str(pth))[0]
    ref, cnd = torch.sqrt(ref), torch.sqrt(cnd)

    chn, out = ref.shape[0], []
    for eig in eig_lst:
        out.append([])
        dif = (cnd[:, :eig] - ref[:, :eig]) / ref[:, :eig]
        dif = dif @ dif.T
        for c in range(chn):
            out[-1].append(dif[c, c].cpu().numpy())
        if chn != 1:
            out[-1].append(torch.trace(dif).cpu().numpy())
    return out


def app_dose(stat, exp, cnd, dct,
             dos, res):
    if cnd not in dct[stat][exp]['one_dos']:
        dct[stat][exp]['one_dos'][cnd] = dict()
    dct[stat][exp]['one_dos'][cnd].update({dos: res})


def min_deig(deig0, deig1):
    for eid in range(len(deig0)):
        for chn in range(len(deig0[eid])):
            deig0[eid][chn] = np.minimum(deig0[eid][chn], deig1[eid][chn])
    return deig0


def acc_deig(stat, exp, cnd, dct, res):
    if cnd in dct[stat][exp]['max_dos']:
        res = min_deig(res,
                       dct[stat][exp]['max_dos'][cnd])
    dct[stat][exp]['max_dos'][cnd] = res


def set_res_plot(stat, exp, cnd,
                 dct, res, res_all):
    dct[stat][exp]['max_dos'][cnd] = res
    dct[stat]['all']['max_dos'][cnd] = res_all

    dct[stat][exp]['all_dos'][cnd] = res
    dct[stat]['all']['all_dos'][cnd] = res_all

    dct[stat][exp]['one_dos'][cnd] = res
    dct[stat]['all']['one_dos'][cnd] = res_all


def set_vis_best(pth, drg_dct):
    # this is meant for remove the redundant pt name for set_vis_dict
    for fld in Path(pth).iterdir():
        for pt in Path(fld).rglob('*.pt'):
            os.rename(str(pt), str(pt).replace(fld.name + '-', ''))

    # select the best stat for each drug
    for key, val in drg_dct.items():
        pth_drg = Path(pth) / key
        kid, avg, scm = [], None, None
        for exp in (1, 2):
            pth_kid = pth_drg / f'{exp}-{val}_kid.pt'
            if pth_kid.is_file():
                kid.append(torch.load(pth_kid))
                print(key, val, exp, 'kid')

            pth_scm = pth_drg / f'{exp}-{val}_scm.pt'
            pth_avg = pth_drg / f'{exp}-{val}_mean.pt'
            if pth_scm.is_file():
                if exp == 1:
                    scm = torch.load(pth_scm)
                    avg = torch.load(pth_avg)
                    print(key, val, exp)
                else:
                    if scm is not None:
                        scm = torch.load(pth_drg / f'{val}_scm.pt')
                        avg = torch.load(pth_drg / f'{val}_mean.pt')
                        print(key, val, exp, '1 exist')
                    else:
                        scm = torch.load(pth_scm)
                        avg = torch.load(pth_avg)
                        print(key, val, exp, '1 does not exist')
        kid = torch.cat(kid, dim=1)
        print(kid.shape, scm[0].shape, scm[1].shape, avg.shape)
        torch.save(kid, pth_drg / f'best_kid.pt')
        torch.save(scm, pth_drg / f'best_scm.pt')
        torch.save(avg, pth_drg / f'best_mean.pt')


def set_vis_dict(pth, dos='', num=None):
    val, vec = torch.load(str(pth / f'{dos}scm.pt'))
    avg = torch.load(str(pth / f'{dos}mean.pt'))
    cod = torch.load(str(pth / f'{dos}kid.pt'))
    if isinstance(num, int) and num > 0:
        idx = torch.randperm(cod.shape[1])
        cod = cod[:, idx[:num], :]
    dct = {'val': val.float(), 'vec': vec.float(),
           'avg': avg.float(), 'cod': cod.float()}
    print(pth.name, val.shape, vec.shape, avg.shape, cod.shape)
    return dct


def get_cluster_plot(arg, pth, eid,
                     cnd_num, cnd_dct, out_dct):
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 12}
    plt.rc('font', **font)

    if arg.data_name == 'rxrx19a':
        chn, ref = 5, 'Mock'
        col = ['DNA', 'ER',
               'Actin', 'RNA',
               'Golgi']
        clr = ['#E72F52', '#0D95D0', '#7DC462', '#D44627', '#774FA0']
    else:
        chn, ref = 6, 'healthy'
        col = ['DNA', 'ER',
               'Actin', 'RNA',
               'Mitochondria', 'Golgi']
        clr = ['#E72F52', '#0D95D0', '#7DC462',
               '#D44627', '#EFB743', '#774FA0']

    # chn = 5 if arg.data_name == 'rxrx19a' else 6
    # ref = 'Mock' if arg.data_name == 'rxrx19a' else 'healthy'
    col = [f'c{c}-{e}' for c in range(chn) for e in range(eid)]
    clr = [c for c in clr for _ in range(eid)]

    stat, ref_key = 'scm', '' if arg.data_name == 'rxrx19a' else '-1'
    ref_pth = pth / ref / f'{ref}{ref_key}_{stat}.pt'
    ref_eig = torch.load(ref_pth)[0]
    print(ref, ref_eig.shape)
    for tot in ('max_dos', 'all_dos'):
        if (pth / 'main_cluster' / f'{tot}.csv').is_file():
            df = pd.read_csv(
                str(pth / 'main_cluster' / f'{tot}.csv'), index_col=0)
        else:
            out_plt = dict()
            for cnd, cval in cnd_dct.items():
                if cnd in out_dct[stat]['all'][tot]:
                    out_plt[cnd] = out_dct[stat]['all'][tot][cnd]
            out_rnk = dict(sorted(out_plt.items(),
                                  key=lambda item: item[1][eid - 1][chn]))

            drg_dct = {}
            for did, drg in enumerate(out_rnk.keys()):
                if drg in arg.control:
                    continue
                if did == cnd_num:
                    break

                print(tot, drg)
                drg_pth = pth / drg
                if tot == 'max_dos':
                    drg_val, drg_stat, has_all = 10e8, None, False
                    dose = prep_dose(cnd_dct[drg], ('1', '2'))
                    for exp in ('', '-1', '-2'):
                        if exp and has_all:
                            continue
                        for dos in dose:
                            key_dos = f'{drg}{exp}-{dos}_{stat}.pt'
                            if (drg_pth / key_dos).is_file():
                                out = calc_deig(ref_eig, drg_pth / key_dos)
                                if out[eid - 1][chn] < drg_val:
                                    drg_val = out[eid - 1][chn]
                                    drg_stat = torch.load(
                                        str(drg_pth / key_dos))[0][:, :eid]
                                    print(key_dos)
                                if not exp and not has_all:
                                    has_all = True
                    drg_stat = torch.sqrt(drg_stat) / \
                        torch.sqrt(ref_eig[:, :eid])
                    drg_stat = (drg_stat - 1) * 100
                    drg_dct[drg] = drg_stat.flatten().cpu().numpy()
                elif tot == 'all_dos':
                    if (drg_pth / f'{drg}_{stat}.pt').is_file():
                        key_dos = f'{drg}_{stat}.pt'
                    else:
                        if (drg_pth / f'{drg}-1_{stat}.pt').is_file():
                            key_dos = f'{drg}-1_{stat}.pt'
                        else:
                            key_dos = f'{drg}-2_{stat}.pt'
                    drg_stat = torch.load(str(drg_pth / key_dos))[0][:, :eid]
                    drg_stat = torch.sqrt(drg_stat) / \
                        torch.sqrt(ref_eig[:, :eid])
                    drg_stat = (drg_stat - 1) * 100
                    drg_dct[drg] = drg_stat.flatten().cpu().numpy()
            df = pd.DataFrame.from_dict(drg_dct, orient='index', columns=col)
            df.to_csv(str(pth / 'main_cluster' / f'{tot}.csv'), index=True)
        sns.clustermap(df,
                       method='weighted',
                       metric='correlation',
                       standard_scale=1,
                       center=0,
                       col_colors=clr,
                       col_cluster=False,
                       yticklabels=True, xticklabels=False)
        plt.savefig(str(pth / 'main_cluster' / f'{tot}_{cnd_num}.png'),
                    bbox_inches='tight', dpi=600)
        plt.close()


def get_base_plot(arg, pth, log,
                  out_dct, hit_dct,
                  eig_scl=100,
                  eig_lst=[1, 2, 3, 4, 5, 10]):
    lfont = {'fontname': 'Helvetica', 'fontsize': 'larger'}
    ref = 'Mock' if arg.data_name == 'rxrx19a' else 'healthy'
    cut_off = 'Infected' if arg.data_name == 'rxrx19a' else 'storm-severe'
    if arg.data_name == 'rxrx19a':
        ref_ = 'Irradiated'

    eig_len = len(out_dct[ref])
    chn_num = len(out_dct[ref][0])
    log.info(str(pth) + f' eig_len:{eig_len}, chn_num:{chn_num}\n')

    ann_cel = ('GS-441524', 'Remdesivir (GS-5734)',
               'Chloroquine', 'Hydroxychloroquine Sulfate', 'Aloxistatin')
    flt_cel = ('',)
    if arg.data_cell == 'HUVEC':
        ann_cel = ('GS-441524', 'Remdesivir (GS-5734)',
                   'Crizotinib', 'Golvatinib', 'Cabozantinib')
    topk = 1
    for chn in range(chn_num):
        if chn != chn_num - 1:
            continue
        for eid, est in enumerate(eig_lst):
            # if est != 5:
            #     continue
            df_hit = pd.DataFrame(hit_dct.items())
            df_hit['ours'] = 'negative'
            df_hit['Proposed'] = None
            df_hit['Hit score (Cuccarese et al.)'] = df_hit[1]

            hit_rnk = dict(sorted(out_dct.items(),
                                  key=lambda item: item[1][eid][chn]))
            msg = f'Hit_score ({chn}-{est}): '
            y_ticks = []
            baseline, irradiated, is_pos = 0, None, True
            for cnd, res in hit_rnk.items():
                dis = float(res[eid][chn] * eig_scl)
                msg += f'{cnd}: {dis} '

                if cnd in arg.control:
                    if cnd == cut_off:
                        baseline, is_pos = dis, False
                        msg += '\n\n'
                        y_ticks.append(baseline)
                    elif cnd == 'Irradiated':
                        irradiated = dis
                        y_ticks.append(irradiated)
                else:
                    df_hit.loc[df_hit[0] == cnd, 'Proposed'] = dis
                    if is_pos:
                        df_hit.loc[df_hit[0] == cnd, 'ours'] = 'positive'
            log.info(msg)

            # Prepare data
            df_hit['color'] = 'royalblue'
            df_hit.loc[df_hit['ours'] == 'negative', 'color'] = 'r'
            df_hit['Proposed'] = df_hit['Proposed'].astype(float)
            df_hit['Hit score (Cuccarese et al.)'] = df_hit['Hit score (Cuccarese et al.)'].astype(
                float)

            y_min = 0
            y_max = np.max((np.max(df_hit['Proposed']), baseline))
            power = 10 ** np.floor(np.log10(y_max))
            y_max = (y_max // power + 1) * power
            y_max = min(y_max, 5 * eig_scl)
            if arg.data_cell == 'VERO':
                y_max = 100 * eig_scl

            x_min = np.min(df_hit['Hit score (Cuccarese et al.)'])
            x_max = 1.0

            # Draw scatter plot
            plt.figure(figsize=(6, 6), dpi=600)
            for label in ('positive', 'negative'):
                df_lab = df_hit[df_hit['ours'] == label]
                scatter_dct = {'color': list(df_lab['color'].values)}
                if arg.data_cell == 'VERO':
                    scatter_dct.update({'s': 64})
                else:
                    scatter_dct.update({'s': 32})
                sns.regplot(data=df_lab, x='Hit score (Cuccarese et al.)', y='Proposed', fit_reg=False,
                            scatter_kws=scatter_dct, label=label)

            df_hit = df_hit.sort_values('Proposed', ascending=False)
            df_hit = df_hit.reset_index(drop=True)
            df_text, tp_rank = [], []
            for i in range(topk):
                fdct = dict(size=14, fontweight='bold')
                if df_hit['Proposed'][i] <= y_max and df_hit[0][i] not in ann_cel:
                    p_text = plt.text(x=df_hit['Hit score (Cuccarese et al.)'][i], y=df_hit['Proposed'][i], s=df_hit[0][i],
                                      fontdict=fdct)
                    df_text.append(p_text)
                    tp_rank.append(df_hit[0][i])

            df_hit = df_hit.sort_values(
                'Hit score (Cuccarese et al.)', ascending=True)
            df_hit = df_hit.reset_index(drop=True)
            for i in range(topk):
                fdct = dict(size=14, fontweight='bold')
                if df_hit['Proposed'][i] <= y_max and \
                   df_hit[0][i] not in ann_cel and \
                   df_hit[0][i] not in tp_rank:
                    p_text = plt.text(x=df_hit['Hit score (Cuccarese et al.)'][i], y=df_hit['Proposed'][i], s=df_hit[0][i],
                                      fontdict=fdct)
                    df_text.append(p_text)

            for cel in ann_cel:
                if cel in df_hit[0].values:
                    hit = df_hit.loc[df_hit[0] == cel,
                                     'Hit score (Cuccarese et al.)'].values
                    our = df_hit.loc[df_hit[0] == cel, 'Proposed'].values
                    fdct = dict(size=14, fontweight='bold')
                    p_text = plt.text(x=hit, y=our, s=cel,
                                      fontdict=fdct)
                    df_text.append(p_text)

            # Draw regression curve
            df_hit = df_hit[~df_hit[0].isin(flt_cel)]
            hit_score = df_hit['Hit score (Cuccarese et al.)'].values
            proposed = df_hit['Proposed'].values
            eval_x = np.linspace(np.min(hit_score),
                                 np.max(hit_score), len(df_hit))
            smoothed, bottom, top = lowess_with_confidence_bounds(
                hit_score, proposed, eval_x, lowess_kw={'frac': 3/4})

            plt.plot(eval_x, smoothed, color='seagreen')
            plt.fill_between(eval_x, bottom, top, alpha=1/3, color='seagreen')

            plt.yscale('symlog')
            plt.ylim(y_min, y_max)
            plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.1e}'))
            plt.yticks(np.sort(list(set(y_ticks))))

            plt.xlim(x_min, x_max)
            xticks = plt.gca().xaxis.get_major_ticks()
            xticks[-1].label1.set_visible(False)
            # plt.xticks(fontsize=8)
            plt.axhline(y=baseline, color='orangered')
            plt.text(x=1.02, y=baseline, s=cut_off, verticalalignment='bottom',
                     fontdict=dict(color='orangered', fontweight='bold'))
            plt.axhline(y=0, color='black')
            plt.text(x=1.02, y=0, s=ref, verticalalignment='top',
                     fontdict=dict(color='black', fontweight='bold'))
            if irradiated:
                plt.axhline(y=irradiated, color='blue')
                plt.text(x=1.02, y=irradiated, s='Irradiated', verticalalignment='bottom',
                         fontdict=dict(color='blue', fontweight='bold'))
            # plt.legend(fontsize=6)
            plt.legend(loc='lower center',
                       bbox_to_anchor=(.5, 1.1), frameon=False)
            adjust_text(df_text,
                        arrowprops=dict(arrowstyle='->', color='black'))
            plt.savefig(str(pth / f'chn{chn}-eig{est}_reg.png'),
                        bbox_inches='tight')
            plt.close()

            # Draw violinplot
            plt.figure(figsize=(6, 4), dpi=600)
            sns.violinplot(data=df_hit, x='Hit score (Cuccarese et al.)', y='ours',
                           order=['negative', 'positive'], cut=0,
                           palette={'negative': 'r', 'positive': 'royalblue'})
            plt.savefig(str(pth / f'chn{chn}-eig{est}_vln.png'),
                        bbox_inches='tight')
            plt.close()


def get_drug_plot(arg, pth,
                  plt_dct,
                  out_dct,
                  eig_scl=100,
                  eig_lst=[1, 2, 3, 4, 5, 10],
                  dos_lst=[0.3, 1.0, 10.0, 30.0]):

    ref = 'Mock' if arg.data_name == 'rxrx19a' else 'healthy'
    cut_off = 'Infected' if arg.data_name == 'rxrx19a' else 'storm-severe'
    if arg.data_name == 'rxrx19a':
        ref_ = 'Irradiated'
        chn_nm = ['DNA', 'Endoplasmic reticulum',
                  'Actin', 'Nucleoli and cytoplasmic RNA',
                  'Golgi and plasma membrane', 'Total']
    else:
        chn_nm = ['DNA', 'Endoplasmic reticulum',
                  'Actin', 'Nucleoli and cytoplasmic RNA',
                  'Mitochondria', 'Golgi and plasma membrane', 'Total']

    for chn in range(len(out_dct[ref][0])):
        for eid, est in enumerate(eig_lst):
            plt.figure(figsize=(6, 6), dpi=600)

            x_max, x_ticks = 0, set()
            y_max = out_dct[cut_off][eid][chn][0]
            y_ticks = [y_max * eig_scl]
            for cnd, res in out_dct.items():
                if cnd not in arg.control:
                    print(cnd, chn, eid)
                    dr_dct = {float(key): val[eid][chn] * eig_scl
                              for key, val in res.items()}
                    df_hit = pd.DataFrame.from_dict(dr_dct, orient='index')
                    x_axis = df_hit.index
                    x_ticks.update(df_hit.index.tolist())
                    df_hit['Concentration'] = df_hit.index
                    df_hit[chn_nm[chn]] = df_hit[0]
                    y_max = max(np.max(df_hit[chn_nm[chn]]),
                                y_max)
                    x_max = max(np.max(list(dr_dct.keys())),
                                x_max)

                    # Draw concentration plot
                    l_dct = {'data': df_hit,
                             'x': 'Concentration',
                             'y': chn_nm[chn], 'label': cnd, 'markersize': 12,
                             'color': plt_dct[cnd][0], 'marker': plt_dct[cnd][1]}
                    ax = sns.lineplot(**l_dct)
                    ax.lines[-1].set_linestyle(plt_dct[cnd][2])
                    if df_hit[1].any() and df_hit[2].any():
                        plt.fill_between(
                            df_hit.index, df_hit[1], df_hit[2], color=plt_dct[cnd][0], alpha=1/5)
            ax.set(ylabel=None, title=chn_nm[chn])
            plt.xlim(0, x_max)
            plt.xscale('symlog')
            x_ticks = np.sort(list(x_ticks))
            x_texts = ['' if x not in dos_lst else x for x in x_ticks]
            plt.xticks(x_ticks, x_texts, rotation=15)

            if arg.data_name == 'rxrx19a':
                y_ticks.append(out_dct[ref_][eid][chn][0] * eig_scl)
            y_ticks = np.sort(list(set(y_ticks)))
            power = 10 ** np.floor(np.log10(y_max))
            y_max = (y_max // power + 1) * power
            plt.ylim(0, y_max)
            plt.yscale('symlog')
            plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.1e}'))
            plt.yticks(y_ticks)

            align = ('top', 'bottom', 'bottom')
            for cid, cnd in enumerate(arg.control.keys()):
                if cnd in ('Mock', 'healthy'):
                    y_val = 0
                else:
                    y_val = out_dct[cnd][eid][chn][0] * eig_scl
                clr = plt_dct[cnd][0]
                plt.axhline(y=y_val, color=clr)
                plt.text(x=x_max * 1.01, y=y_val, s=cnd, verticalalignment=align[cid],
                         fontdict=dict(color=clr))

                if cnd not in ('Mock', 'healthy'):
                    lower, upper = out_dct[cnd][eid][chn][1], out_dct[cnd][eid][chn][2]
                    if lower and upper:
                        x_axis = np.linspace(0, x_max, 100)
                        plt.fill_between(x_axis,
                                         [lower * eig_scl] * 100,
                                         [upper * eig_scl] * 100,
                                         color=clr, alpha=1/5)

            plt.legend(loc='lower center',
                       bbox_to_anchor=(.5, 1.1), frameon=False)
            plt.savefig(str(pth / f'chn{chn}-eig{est}_conc.png'),
                        bbox_inches='tight')
            plt.close()


def get_pca_plot(pth, chn, col, slc,
                 cnd_dct, plt_dct,
                 kct_df=None,
                 red_dim='PC'):

    if 'Infected' in list(cnd_dct.keys()):
        chn_nm = ['DNA', 'Endoplasmic reticulum',
                  'Actin', 'Nucleoli and cytoplasmic RNA',
                  'Golgi and plasma membrane', 'Total']
    elif 'storm-severe' in list(cnd_dct.keys()):
        chn_nm = ['DNA', 'Endoplasmic reticulum',
                  'Actin', 'Nucleoli and cytoplasmic RNA',
                  'Mitochondria', 'Golgi and plasma membrane', 'Total']
    else:
        chn_nm = None

    df, qy, qx, palette = [], None, None, {}
    for cnd, cod in cnd_dct.items():
        print(cnd)
        if cnd in ('Infected', 'storm-severe'):
            qx = torch.quantile(cod[chn, :, slc[0]], 1.)
            qy = torch.quantile(cod[chn, :, slc[1]], 0.99)
        if cnd == 'nv':
            qx = torch.quantile(cod[chn, :, slc[0]], 1.)
            qy = torch.quantile(cod[chn, :, slc[1]], 1.)
        palette.update({cnd: plt_dct[cnd][0]})
        pca_df = pd.DataFrame(data=cod[chn, :, slc].numpy(),
                              columns=col)
        pca_df['Condition'] = cnd
        df.append(pca_df)
    df_all = pd.concat(df, ignore_index=True)
    print(df_all, col[0], col[1])
    g = sns.jointplot(data=df_all, alpha=0, kind='kde', fill=True,
                      x=col[0], y=col[1],
                      hue='Condition', palette=palette,
                      marginal_kws={'common_norm': False})
    if chn_nm is not None:
        sns.move_legend(g.ax_joint, 'lower center',
                        bbox_to_anchor=(.5, 1.3), frameon=False)
    for cnd in cnd_dct:
        cnd_df = df_all[df_all['Condition'] == cnd]
        fill, level = True, 5
        if cnd in ('healthy', 'Mock', 'nv', 'Infected', 'storm-severe'):
            fill, level = False, 5
        sns.kdeplot(data=cnd_df, ax=g.ax_joint,
                    x=col[0], y=col[1], thresh=0.02,
                    color=plt_dct[cnd][0], alpha=0.8,
                    fill=fill, levels=level)

    # if kct_df is not None:
    #     scatter_dct = {'color': list(kct_df['color'].values)}
    #     rplt = sns.regplot(data=kct_df, x=col[0], y=col[1], fit_reg=False, ax=g.ax_joint,
    #                        scatter_kws=scatter_dct)

    #     fdct = dict(size=8, fontweight='bold')
    #     for index, row in kct_df.iterrows():
    #         rplt.text(x=row[col[0]], y=row[col[1]], s=index + 1,
    #                   fontdict=fdct)

    if red_dim == 'PC' and qx is not None:
        g.ax_joint.set_xlim(right=qx.numpy())
        g.ax_joint.set_ylim(top=qy.numpy())
    if chn_nm is not None:
        plt.suptitle(chn_nm[chn], y=1)
    plt.savefig(str(pth / f'chn{chn}_{red_dim}.png'),
                bbox_inches='tight', dpi=600)
    plt.close()


def get_pca_res(col, cod, pca, ref,
                pca_cls=2,
                pca_num=16):
    if len(ref.shape) == 3:
        c, _, p = ref.shape
        pca = pca.reshape(-1, p*c)
        print('ref', ref.shape)

    cod_dct = {}
    print('pca', pca.shape)
    kmeans = KMeans(n_clusters=pca_cls, random_state=0).fit(pca)
    for kct in kmeans.cluster_centers_:
        # compute l2 distance to centroid
        dist = ((pca - kct)**2).sum(axis=1)
        didx = dist.argsort()[:pca_num]
        # for each list of latent representations
        # extract pca_num ones that are closest to centroid
        lab = kmeans.labels_[didx[0]]
        cod_dct[lab] = cod[:, didx].float()

    kct_np = np.asarray(kmeans.cluster_centers_)
    if len(ref.shape) == 3:
        kct_np = kct_np.reshape(c, -1, p)
    kct_np = kct_np ** 2 / ref

    if len(ref.shape) == 3:
        kct_np = np.sum(kct_np, axis=0)
    assert len(kct_np.shape) == 2
    if kct_np.shape[1] > 2:
        print('kct', kct_np.shape)
        kct_np[:, 1] = np.sum(kct_np[:, 1:], axis=-1)
        kct_np = kct_np[:, :2]
    kct_df = pd.DataFrame(kct_np, columns=col)
    return cod_dct, kct_df


def get_man_img(arg, pth, chn, slc,
                clr, wei, vec,
                cod_dct, model, eig_nm=''):
    for cid, cod in cod_dct.items():
        outlst = list()
        for pow in torch.linspace(-arg.plot_powr, arg.plot_powr, arg.plot_step):
            mancod = run_one_eig_new(arg, arg.plot_base,
                                     pow * wei,
                                     cod, vec)

            output = run_one_dec(arg, model, mancod)
            output = F.interpolate(output.detach(),
                                   size=64,
                                   mode='bilinear')
            outlst.append(output)

        inp = torch.zeros_like(outlst[-1])
        outlst.append(inp)

        nam = 'slc'
        for s in slc:
            nam += f'{s}_'

        save_inf = f'{eig_nm}_{nam}chn{chn}_grp{cid + 1}'
        get_img_plot(outlst,
                     sdir=pth / save_inf,
                     bb_clr=clr,
                     bb_len=2,
                     left=True)


def get_met_res(arg, log, met_key, cnd_dct):
    # calc numerical results: psnr, ssim
    # including all channel stats (met_chn + 1)
    pth = arg.save_path
    for met in met_key:
        for chn in range(arg.img_chn + 1):
            if 'rxrx19' in arg.data_name:
                met_all, met_drg = None, None
                for drg, val in cnd_dct.items():
                    for exp in ('1', '2'):
                        if val[exp] is not None:
                            met_pth = pth / drg / f'{drg}-{exp}_{met}_{chn}.pt'

                    if val['1'] is not None and val['2'] is not None:
                        met_pth = pth / drg / f'{drg}_{met}_{chn}.pt'

                    met_res = torch.load(met_pth)
                    if met_all is not None:
                        met_all = torch.cat([met_all, met_res])
                    else:
                        met_all = met_res

                    if drg in arg.control:
                        msg = gen_msg(met_res, chn, met, drg)
                        log.info(msg)
                    else:
                        if met_drg is not None:
                            met_drg = torch.cat([met_drg, met_res])
                        else:
                            met_drg = met_res

                # exclude huvec very small amount of nan cases
                met_clean = torch.nan_to_num(met_drg)
                msg = gen_msg(met_drg[met_clean == met_drg],
                              chn, met, 'drug of interests')
                log.info(msg)
                met_clean = torch.nan_to_num(met_all)
                msg = gen_msg(met_all[met_clean == met_all], chn, met, 'total')
                log.info(msg)
            else:
                met_all = None
                for cnd, val in cnd_dct.items():
                    met_pth = pth / cnd / f'{met}_{chn}.pt'
                    met_res = torch.load(met_pth)
                    if met_all is not None:
                        met_all = torch.cat([met_all, met_res])
                    else:
                        met_all = met_res

                    msg = gen_msg(met_res, chn, met, cnd)
                    log.info(msg)
                msg = gen_msg(met_all, chn, met, 'total')
                log.info(msg)


def load_kid(pth, skn):
    skn_pth = pth / skn / 'kid.pt'
    skn_out = torch.load(str(skn_pth)).squeeze(0).T
    skn_out = skn_out.double().cpu().numpy()
    return skn_out


def get_dis_res(arg, log, plt_dct):
    if arg.data_name == 'ham10k':
        pth = arg.save_path
        percent = [0, 0.25, 0.5, 0.75, 1.0]

        dis_dct = {}
        ref = load_kid(arg.save_path, 'nv')
        for skn in plt_dct:
            dis_dct[skn] = dict()
            cnd = load_kid(arg.save_path, skn)
            for per in percent:
                calc_toy_measure(ref, cnd, per, log, dis_dct[skn], ' '*4)
        print(dis_dct)

        dis_lst = [key for key in dis_dct['mel'] if 'std' not in key]
        for dis in dis_lst:
            print(dis)
            plt.figure(figsize=(6, 6), dpi=600)
            out_dct = {key: val[dis] for key, val in dis_dct.items()}
            std_dct = {key: val[f'{dis}_std']
                       for key, val in dis_dct.items()}
            out_df = pd.DataFrame.from_dict(out_dct)
            out_df['Interpolation weight'] = percent

            std_df = pd.DataFrame.from_dict(std_dct)
            std_df['Interpolation weight'] = percent

            for skn in plt_dct:
                l_dct = {'data': out_df,
                         'x': 'Interpolation weight',
                         'y': skn, 'label': skn, 'markersize': 12,
                         'color': plt_dct[skn][0], 'marker': plt_dct[skn][1]}
                ax = sns.lineplot(**l_dct)
                ax.lines[-1].set_linestyle(plt_dct[skn][2])
                ax.set(ylabel=None)
                lower = out_df[skn] - std_df[skn]
                upper = out_df[skn] + std_df[skn]
                if '_test' in dis:
                    upper = np.minimum(upper, 1.)
                plt.fill_between(std_df['Interpolation weight'],
                                 lower, upper, color=plt_dct[skn][0], alpha=1/5)

            plt.legend()
            plt.xlim(0, 1)
            plt.xticks(percent, percent)

            if '_test' in dis:
                plt.ylim(0, 1)
                y_ticks = [0.2, 0.4, 0.6, 0.8, 1.0]
                plt.yticks(y_ticks, y_ticks)
            else:
                plt.yscale('symlog')
                if 'L_2' in dis or 'Proposed' in dis:
                    plt.gca().set_ylim(bottom=0)
                if 'Proposed' in dis:
                    plt.yticks([0, 0.01, 0.1])
                # else:
                #     plt.yticks(fontsize=8)
            pth = arg.save_path / 'main_quant'
            pth.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(pth / f'{dis}.png'))
            plt.close()
    else:
        pass


def _d_novel(sigma1, sigma2):
    val = eigvals(sigma1 @ sigma2)
    val = val.real
    val[val < 0] = 0
    return 2 * np.sqrt(val).sum()


def calc_fid(mu1, mu2, sigma1, sigma2):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2
    fid_easy = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2)
    fid_hard = _d_novel(sigma1, sigma2)
    fid = fid_easy - fid_hard
    return fid


def calc_kid(cmp0, cmp1, num_subsets=100, max_subset_size=1000):
    """ Refactor based on
    https://github.com/NVlabs/stylegan3/blob/main/metrics/kernel_inception_distance.py

    Args:
        args: arguments that are implemented in args.py file
            such as data_name, data_splt.
    """

    n = cmp0.shape[1]
    m = min(min(cmp0.shape[0], cmp1.shape[0]), max_subset_size)
    t = 0.
    for i in range(num_subsets):
        # if (i + 1) % 20 == 0:
        #     print('kid', i, n)
        x = cmp0[np.random.choice(cmp0.shape[0], m, replace=False)]
        y = cmp1[np.random.choice(cmp1.shape[0], m, replace=False)]
        a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
        b = (x @ y.T / n + 1) ** 3
        t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
    kid = t / num_subsets / m
    return kid


def calc_pval(cmp0, cmp1, log, off_set):
    ks = ks_2samp(cmp0, cmp1)
    msg_ks = '{}KS_test: {}, {}'.format(off_set, ks.statistic, ks.pvalue)
    log.info(msg_ks)

    tt = ttest_ind(cmp0, cmp1, equal_var=False)
    msg_tt = '{}T_test: {}, {}'.format(off_set, tt.statistic, tt.pvalue)
    log.info(msg_tt)

    wl = mannwhitneyu(cmp0, cmp1)
    msg_wl = '{}Wilcoxon_test: {}, {}'.format(
        off_set, wl.statistic, wl.pvalue)
    log.info(msg_wl)

    lv = levene(cmp0, cmp1)
    msg_lv = '{}Levene_test: {}, {}'.format(
        off_set, lv.statistic, lv.pvalue)
    log.info(msg_lv)

    F = np.var(cmp0, ddof=1) / np.var(cmp1, ddof=1)
    ft = f.cdf(F, len(cmp0)-1, len(cmp1)-1)
    msg_ft = '{}F_test: {}'.format(off_set, ft)
    log.info(msg_ft)

    return (ks.pvalue, tt.pvalue, wl.pvalue, lv.pvalue, ft)


def calc_l2(ref, cnd, log, off_set):
    ref, cnd = np.sqrt(ref), np.sqrt(cnd)
    dif = ref - cnd
    l2 = np.sqrt(dif.dot(dif))
    msg = f'{off_set}L_2: {l2}'
    log.info(msg)
    return l2


def calc_prop(ref, cnd, log, off_set):
    ref, cnd = np.sqrt(ref), np.sqrt(cnd)
    out_lst = []
    for r in [1, 2, 3, 4, 5]:
        dif = (ref[:r] - cnd[:r]) / ref[:r]
        out = np.sqrt(dif.dot(dif))
        out_lst.append(out)
        msg = f'{off_set}Proposed_{r}: {out}'
        log.info(msg)
    return out_lst


def calc_toy_measure(ref, cnd,
                     per, log,
                     dis_dct,
                     off_set):
    off_set_next = off_set + off_set

    ref_len, cnd_len = ref.shape[1], cnd.shape[1]
    ref_per, cnd_per = int(ref_len * per), int(cnd_len * (1 - per))

    kid, l2 = [], []
    prop_lst = [[] for _ in range(5)]
    pval_lst = [[] for _ in range(5)]
    for s in range(4):
        np.random.seed(s)
        if per == 0:
            cmp = cnd
        elif per == 1.0:
            cmp = ref
        else:
            cmp = np.concatenate([ref[:, np.random.choice(ref_len, ref_per, replace=False)],
                                  cnd[:, np.random.choice(cnd_len, cnd_per, replace=False)]], axis=1)
        print(s, ref.shape, cmp.shape)
        kid.append(calc_kid(ref.T, cmp.T))
        if s > 0 and per in (0, 1.0):
            continue
        ref_val = svdvals((ref @ ref.T) / ref.shape[1])
        cmp_val = svdvals((cmp @ cmp.T) / cmp.shape[1])
        l2.append(calc_l2(ref_val, cmp_val, log,
                          off_set_next))

        prop = calc_prop(ref_val, cmp_val, log,
                         off_set)
        for i in range(5):
            prop_lst[i].append(prop[i])

        pval = calc_pval(ref_val, cmp_val, log,
                         off_set_next)
        for i in range(5):
            pval_lst[i].append(pval[i])

    save_dct('KID', np.mean(kid), dis_dct)
    save_dct('KID_std', np.std(kid), dis_dct)
    log.info(f'{off_set}KID: {np.mean(kid)}, {np.std(kid)} \n')

    save_dct('L_2', np.mean(l2), dis_dct)
    save_dct('L_2_std', np.std(l2), dis_dct)
    log.info(f'{off_set}L_2: {np.mean(l2)}, {np.std(l2)} \n')

    stat_test = ['KS_test', 'T_test', 'Wilcoxon_test', 'Levene_test', 'F_test']
    for i in range(5):
        avg, std = np.mean(prop_lst[i]), np.std(prop_lst[i])
        save_dct(f'Proposed_{i + 1}', avg, dis_dct)
        save_dct(f'Proposed_{i + 1}_std', std, dis_dct)
        log.info(f'{off_set}Proposed_{i + 1}: {avg}, {std} \n')
        avg, std = np.mean(pval_lst[i]), np.std(pval_lst[i])
        save_dct(f'{stat_test[i]}', avg, dis_dct)
        save_dct(f'{stat_test[i]}_std', std, dis_dct)
        log.info(f'{stat_test[i]}: {avg}, {std} \n')


def save_dct(key, val, dct):
    if key not in dct:
        dct[key] = [val]
    else:
        dct[key].append(val)


def get_img_recon(input, output, bb_clr, bb_len, sdir):
    plot_dim = list(input.shape)
    plot_dim[0] *= 2
    plot_out = torch.zeros(plot_dim).to(input)

    _nrow = input.shape[0]
    for cid, chn in enumerate(('R', 'G', 'B', 'all')):
        inp, out = input.clone(), output.clone()
        if cid < 3:
            inp = inp[:, cid].unsqueeze(1).repeat(1, 3, 1, 1)
            out = out[:, cid].unsqueeze(1).repeat(1, 3, 1, 1)
        if bb_clr is None:
            plot_out[::2] = inp
        else:
            plot_out[::2] = add_bbx(inp, bb_clr, bb_len)
        plot_out[1::2] = out
        plot_pth = sdir.parent / f'recon_{chn}_{sdir.name}.png'
        tv_utils.save_image(plot_out.float(),
                            str(plot_pth),
                            nrow=2,
                            padding=2)


def get_img_manip(output, bbx_clr, bbx_len, sdir, left):
    _nrow = len(output)

    plot_dim = list(output[0].shape)
    plot_dim[0] *= _nrow
    plot_out = torch.zeros(plot_dim).to(output[0])

    for cid, chn in enumerate(('R', 'G', 'B', 'all')):
        for i in range(_nrow):
            out = output[i].clone()
            if cid < 3:
                out = out[:, cid].unsqueeze(1).repeat(1, 3, 1, 1)
            if i == _nrow // 2 and bbx_clr is not None:
                out = add_bbx(out, bbx_clr, bbx_len)

            if left:
                plot_out[i::_nrow] = out
            else:
                plot_out[_nrow - i - 1::_nrow] = out
        plot_pth = sdir.parent / f'manip_{chn}_{sdir.name}.png'
        tv_utils.save_image(plot_out.float(),
                            str(plot_pth),
                            nrow=_nrow,
                            padding=2)


def get_img_plot(output, sdir, bb_clr, bb_len, axis=1, left=True,
                 is_man=True, is_rec=False):
    for oid, out in enumerate(output):
        if out.shape[axis] == 6:
            out = add_contrast(out, axis)
            # always append 345 channels along rows
            output[oid] = torch.cat([out[:, :3].clone(),
                                     out[:, 3:].clone()], axis=-2)
        elif out.shape[axis] == 5:
            mito = torch.zeros([out.shape[0], 1,
                                out.shape[2], out.shape[3]])
            out = torch.cat([out, mito.to(out)], axis=1)
            # always append 345 channels along rows
            output[oid] = torch.cat([out[:, :3].clone(),
                                     out[:, 3:].clone()], axis=-2)

    if is_rec:
        # assume the reconstruted image is
        # at the middle of the list
        pos = len(output) // 2 - 1
        get_img_recon(output[pos].clone(),
                      output[-1].clone(),
                      bb_clr, bb_len,
                      sdir)
    if is_man:
        # remove the input image
        get_img_manip(output[:-1],
                      bb_clr, bb_len,
                      sdir, left)


def main():
    drg_dct = {'VERO': {'Idelalisib': 10.0, 'Remdesivir (GS-5734)': 10.0, 'GS-441524': 10.0},
               'HRCE': {'Bortezomib': 0.01, 'Remdesivir (GS-5734)': 10.0, 'GS-441524': 10.0},
               'HUVEC': {'Bortezomib': 0.01, 'Crizotinib': 2.5,
                         'Tofacitinib': 3.0, 'Golvatinib': 3.0, 'Cabozantinib': 2.5}}
    for cel in drg_dct:
        print(cel)
        pth = f'Experiment/ldim_visual_strat/{cel}_visual/psp_style2_800000_False_False_False_0_True/'
        set_vis_best(pth, drg_dct[cel])
        print()


if __name__ == '__main__':
    main()
