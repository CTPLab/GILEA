import unittest
import numpy as np
from PIL import Image
from pathlib import Path


class TestHam10k(unittest.TestCase):
    def test_stat(self,
                  size=64,
                  path='non_IID/preproc/ham10k/images'):

        old_pth = Path(path)
        new_pth = path.replace('ham10k', 'ham10k_tiny')
        new_pth = Path(new_pth)

        for pid, pth in enumerate(old_pth.glob('**/*.jpg')):
            if (pid + 1) % 1000 == 0:
                print(pid)
            old_img = Image.open(pth).resize(
                (size, size)).resize((size * 2, size * 2))
            old_img = np.asarray(old_img)
            new_img = str(pth).replace('ham10k', 'ham10k_tiny')
            new_img = new_img.replace('.jpg', '.png')
            new_img = np.asarray(Image.open(new_img))
            np.testing.assert_array_equal(old_img, new_img)


if __name__ == "__main__":
    unittest.main()
