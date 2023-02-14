
import numpy as np
from pathlib import Path
from scipy.linalg import svd, eigh




pth = 'non_IID/exps_rxrx19ab/VERO_more_eigh/psp_style2_800000_False_False_False_0_False/'
num = 5

for im_dir in Path(pth).iterdir():
    if im_dir.name in ('Favipiravir',):
        print(im_dir.name)
        for n in range(num):
            scm_pth = f'codes{n}_scm.npy'
            print(scm_pth)
            scm = np.load(str(im_dir/scm_pth))
            _, eigval1 = svd(scm, full_matrices=False)[:2]
            print(eigval1[:5], eigval1[-5:])
            eigval2, _ = eigh(scm)
            eigval2 = eigval2[::-1]
            print(eigval2[:5], eigval2[-5:])