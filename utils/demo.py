import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image, ImageFont, ImageDraw
from utils.plot import get_img_plot
from utils.common import add_bbx, add_contrast, run_one_enc, run_one_dec, run_one_eig_new


def get_raw_header(cfg, cmp0, cmp1,
                   raw_img, sav_pth, hed_clr, font_pth):
    print('head color', hed_clr)

    # This header annotation may only work for RxRx19b
    raw_out = Image.fromarray(raw_img)
    # raw_txt = f'Manipulation: From {cmp0} to {cmp1}'
    font = ImageFont.truetype(str(font_pth), 20)
    # ImageDraw.Draw(raw_out).text(
    #     (0, 0), raw_txt,
    #     (255, 255, 255), font=font)

    raw_012 = ('Channel        |Hoechst 33342  |Concanavalin A              |Phalloidin \n'
               'Visualization  |DNA                  |Endoplasmic reticulum   |Actin \n'
               'Color             |Red                   |Green                             |Blue')
    if cfg.cell_chn == 6:
        raw_345 = ('Channel        |SYTO 14                                    |MitoTracker Deep Red   |WGA \n'
                   'Visualization  |Nucleoli and cytoplasmic RNA   |Mitochondria                  |Golgi and plasma membrane \n'
                   'Color             |Red                                            |Green                             |Blue')
    else:
        raw_345 = ('Channel        |SYTO 14                                    |WGA \n'
                   'Visualization  |Nucleoli and cytoplasmic RNA   |Golgi and plasma membrane \n'
                   'Color             |Red                                            |Green')
    font = ImageFont.truetype(str(font_pth), 14)
    ImageDraw.Draw(raw_out).text(
        (0, 0), raw_012,
        (255, 255, 255), font=font)
    ImageDraw.Draw(raw_out).text(
        (cfg.crop, 0), raw_345,
        (255, 255, 255), font=font)

    raw_out_pth = sav_pth / f'{cmp0}_{cmp1}_raw.png'
    raw_out.save(str(raw_out_pth))
    return raw_out


def get_raw_img(cfg, ref, cnd, raw_pth, sav_pth, hed_clr, font_pth):

    # This function may only work for RxRx19 raw image
    raw_img = list()
    l = (cfg.size - cfg.crop) // 2
    t = (cfg.size - cfg.crop) // 2
    for chn in range(cfg.cell_chn):
        chn_pth = str(raw_pth).replace(
            '_w1.png', '_w{}.png'.format(chn + 1))
        chn_img = np.asarray(Image.open(chn_pth))
        # center crop the image
        chn_img = chn_img[t:t+cfg.crop, l:l+cfg.crop]

        raw_img.append(chn_img.astype(np.float32))
    if cfg.cell_chn == 5:
        raw_img.append(np.zeros_like(raw_img[-1]))

    raw_img = np.stack(raw_img, -1)
    if cfg.cell_chn == 6:
        raw_img = add_contrast(raw_img / 255., axis=2) * 255.
    for chn in range(raw_img.shape[2]):
        cimg = Image.fromarray(raw_img[:, :, chn].astype(np.uint8))
        cimg = cimg.convert('RGB')
        cimg.save(str(sav_pth / f'chn{chn}.png'))

    raw_out = np.concatenate((raw_img[:, :, :3],
                              raw_img[:, :, 3:]), axis=1)

    raw_out = raw_out.astype(np.uint8)
    raw_out = get_raw_header(cfg, ref, cnd,
                             raw_out,
                             sav_pth,
                             hed_clr,
                             font_pth)
    return np.array(raw_out), raw_img


def get_bbx_img(output,
                bbx_img,
                chn_img,
                row, col,
                dim=64, crop=1024):
    output = F.interpolate(output.detach(),
                           size=dim,
                           mode='bilinear').squeeze()

    shf = dim // 2
    output = output.transpose(0, 2).transpose(0, 1)
    if output.shape[2] == 6:
        output = add_contrast(output, axis=2) * 255
    if output.shape[2] == 5:
        mito = torch.zeros_like(output[:, :, [0]])
        output = torch.cat([output * 255, mito.to(output)], dim=2)
    output = output.cpu().numpy()

    out0 = output[:, :, :3]
    out_012 = add_bbx(out0).astype('uint8')
    bbx_img[row-shf:row+shf,
            col-shf:col+shf] = out_012

    out1 = output[:, :, 3:]
    out_345 = add_bbx(out1).astype('uint8')
    bbx_img[row-shf:row+shf,
            col-shf+crop:col+shf+crop] = out_345

    chn_012 = add_bbx(chn_img[row-shf:row+shf,
                              col-shf:col+shf:, :3]).astype('uint8')
    chn_img[row-shf:row+shf,
            col-shf:col+shf:, :3] = chn_012

    chn_345 = add_bbx(chn_img[row-shf:row+shf,
                              col-shf:col+shf:, 3:]).astype('uint8')
    chn_img[row-shf:row+shf,
            col-shf:col+shf:, 3:] = chn_345


def get_vid_img(arg, img,
                wei, vec, dim,
                model, avgim, writers,
                config=None):
    #     pos, msk,
    # crop, bbx_img, chn_img,
    # ):
    shf, out_lst, inp_lst = dim // 2, [], []
    for pow in np.linspace(-arg.demo_powr, arg.demo_powr, arg.demo_step):
        pow_lst = []
        print(pow)
        if arg.data_name == 'ham10k':
            bbx_img = None
        elif 'rxrx19' in arg.data_name:
            assert config is not None
            bbx_img, chn_img = config.raw_img, config.chn_img
            msk = np.zeros((bbx_img.shape[0],
                           bbx_img.shape[1]))
            msk[:64] = 1
            msk[-64:] = 1

        cnt = 0
        for n in range(img.shape[0]):
            if cnt == arg.demo_cell:
                break

            if 'rxrx19' in arg.data_name:
                # ['AC11_s1_454_115.png']
                cur = config.pos[n][0].split('_')
                # 454, '115.png'
                row, col = int(cur[2]), int(cur[3].split('.')[0])
                # avoid cell overlap
                if not np.all(msk[row-shf:row+shf, col-shf:col+shf] == 0.):
                    continue
                msk[row-shf:row+shf, col-shf:col+shf] = 1
            cnt += 1

            codes = run_one_enc(arg, model,
                                avgim, img[n].unsqueeze(0))
            codes = torch.stack(codes)
            if not arg.is_total:
                codes = codes[:, :, :, :128]
            codes = run_one_eig_new(arg, arg.demo_base, wei * pow, codes, vec)
            output = run_one_dec(arg, model, codes)

            if pow in (-arg.demo_powr, -arg.demo_powr / 2,  0, arg.demo_powr / 2, arg.demo_powr):
                if pow == -arg.demo_powr:
                    inp_lst.append(img[n].unsqueeze(0).float().div(255.))
                pow_lst.append(output)

            if arg.data_name == 'ham10k':
                output = F.interpolate(output.detach(),
                                       size=dim,
                                       mode='bilinear').squeeze()
                output = output.transpose(0, 2).transpose(0, 1)
                output = (output * 255).cpu().numpy()
                output = output.astype('uint8')
                if bbx_img is None:
                    bbx_img = output
                else:
                    bbx_img = np.concatenate([bbx_img, output],
                                             axis=1)
            elif 'rxrx19' in arg.data_name:
                get_bbx_img(output,
                            bbx_img, chn_img,
                            row, col, dim, config.crop)

        if pow in (-arg.demo_powr, -arg.demo_powr / 2,  0, arg.demo_powr / 2, arg.demo_powr):
            if pow == -arg.demo_powr:
                inp_lst = torch.cat(inp_lst)
                if 'rxrx19' in arg.data_name:
                    for chn in range(chn_img.shape[-1]):
                        cimg = Image.fromarray(chn_img[:, :, chn])
                        cimg = cimg.convert('RGB')
                        cpth = arg.save_path / \
                            'main_demo' / f'chn{chn}_bbx.png'
                        cimg.save(str(cpth))
            out_lst.append(torch.cat(pow_lst))

        for wid, wrt in enumerate(writers):
            if wid < 3:
                wrt.append_data(bbx_img[:, :, wid])
            else:
                wrt.append_data(bbx_img)
    print(out_lst[-1].shape, inp_lst.shape)
    out_lst.append(inp_lst)
    get_img_plot(out_lst,
                 sdir=arg.save_path / 'main_demo' / 'fig1',
                 bb_clr=None,
                 bb_len=2,
                 left=True,
                 is_rec=True)
