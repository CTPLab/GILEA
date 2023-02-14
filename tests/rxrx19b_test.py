import random
import unittest
import numpy as np
from PIL import Image
from pathlib import Path


class TestRxRx19b(unittest.TestCase):
    def test_stat(self,
                  size=512,
                  crop=32,
                  path='non_IID_old/preproc/rxrx19b/images/HUVEC-1'):

        old_pth = Path(path)
        new_pth = path.replace('rxrx19b', 'rxrx19b_cell')
        new_pth = Path(new_pth)

        imgs_dict = dict()
        for pth in new_pth.glob('**/*.png'):
            plate = pth.parent.name
            if plate not in imgs_dict:
                imgs_dict[plate] = dict()
            # AC46_s1_37_74 -> [AC46 s1 37 74]
            img_info = pth.stem.split('_')
            # AC46_s1
            img_name = img_info[0] + '_' + img_info[1]
            print(pth)
            if img_name not in imgs_dict[plate]:
                imgs_dict[plate][img_name] = set()
            img_cord = img_info[2] + '_' + img_info[3]
            imgs_dict[plate][img_name].add(img_cord)

        plates = list(imgs_dict.keys())
        random.shuffle(plates)
        for plate in plates:
            print(plate)
            imgs = list(imgs_dict[plate].keys())
            random.shuffle(imgs)
            for img in imgs:
                raw_img = list()
                for i in range(6):
                    im = Image.open(old_pth / plate /
                                    '{}_w{}.png'.format(img, i + 1))
                    raw_img.append(im)
                print(img, len(raw_img))

                for img_cord in imgs_dict[plate][img]:
                    row, col = img_cord.split('_')
                    im = Image.open(new_pth / plate /
                                    '{}_{}_{}.png'.format(img, row, col))
                    im = np.array(im)
                    l = size + int(col) - crop
                    t = size + int(row) - crop
                    for i in range(6):
                        raw_crop = raw_img[i].crop((l, t,
                                                    l + crop * 2, t + crop * 2))
                        raw_crop = raw_crop.resize([crop * 4, crop * 4])
                        raw_crop = np.array(raw_crop)
                        new_crop = im[:,
                                      crop * 4 * (i // 3):crop * 4 * (i // 3 + 1),
                                      i % 3]
                        np.testing.assert_array_equal(raw_crop, new_crop)


if __name__ == "__main__":
    unittest.main()
