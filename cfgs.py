class Config(object):
    def __init__(self):
        self.rxrx19 = dict()
        self.rxrx19['VERO'] = {'size': 1024,  # raw image size
                               'crop': 768,  # remaining image size after cropping
                               'sigma': 4,   # sigma for gaussian_filter to process raw images for counting cells
                               'thres': 10,  # the color intensity of a valid cell's centriod must >= thres
                               # the amount of cells segmented from each raw image, cell_num = 200 if ablation
                               'cell_num': 100,
                               'cell_chn': 5,  # the amount of cell image channel
                               # half of the cell image dimension (32 * 2 = 64)
                               'cell_dim': 32,
                               # crop cells within the spatial dim (cell_edg, cell_edg, crop-cell_edg, crop-cell_edg)
                               'cell_edg': 32,
                               'cell_buf': 50000,  # the buffer batch size stored for KID and clustering computation,
                               'control': {'Mock': 'Mock',
                                           'Irradiated': 'UV Inactivated SARS-CoV-2',
                                           'Infected': 'Active SARS-CoV-2'}
                               }
        self.rxrx19['HRCE'] = {'size': 1024,
                               'crop': 768,
                               'sigma': 9,
                               'thres': 10,
                               'cell_num': 50,  # cell_num = 100 if ablation
                               'cell_chn': 5,
                               'cell_dim': 32,
                               'cell_edg': 32,
                               'cell_buf': 50000,
                               'control': {'Mock': 'Mock',
                                           'Irradiated': 'UV Inactivated SARS-CoV-2',
                                           'Infected': 'Active SARS-CoV-2'}
                               }
        # some args may tuned directly in prep_rxrx19b.py
        self.rxrx19['HUVEC'] = {'size': 2048,
                                'crop': 1024,
                                'sigma': 8,
                                'thres': 12,
                                'cell_num': 200,  # cell_num = 300 if ablation,
                                'cell_chn': 6,
                                'cell_dim': 32,
                                'cell_edg': 32,
                                'cell_buf': 50000,
                                'control': {'healthy': 'healthy',
                                            'storm-severe': 'storm-severe'}
                                }
        self.ham10k = dict()
