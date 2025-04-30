# Auto-Denoise

[![ci](https://github.com/CEA-MetroCarac/auto-denoise/workflows/Python%20package/badge.svg)](https://github.com/CEA-MetroCarac/auto-denoise/actions/workflows/ci.yml?query=workflow%3APython)
[![documentation](https://img.shields.io/badge/docs-mkdocs%20material-blue.svg?style=flat)](https://CEA-MetroCarac.github.io/auto-denoise/)
[![pypi version](https://img.shields.io/pypi/v/auto-denoise.svg)](https://pypi.org/project/auto-denoise/)
[![gitter](https://badges.gitter.im/join%20chat.svg)](https://gitter.im/auto-denoise/community)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Auto-denoise (autoden) provides implementations for a small selection of unsupervised and self-supervised CNN denoising methods.
These methods currently include:

* Noise2Noise (N2N) - A self-supervised denoising method using pairs of images of the same object [1].
* Noise2Void (N2V) - A self-supervised denoising method capable of working with a single image [2]. We have also implemented a later development of the method that can work with structured noise [3].
* Deep Image Prior (DIP) - An unsupervised denoising/upsampling/deconvolution method that can also work with a single image [4].

We also provide example implementations of supervised denoising methods, and the tomography specific Noise2Inverse (N2I) method [5].

References:

- [1] J. Lehtinen et al., “Noise2Noise: Learning Image Restoration without Clean Data,” in Proceedings of the 35th International Conference on Machine Learning, J. Dy and A. Krause, Eds., in Proceedings of Machine Learning Research, vol. 80. PMLR, 2018, pp. 2965–2974. https://proceedings.mlr.press/v80/lehtinen18a.html
- [2] A. Krull, T.-O. Buchholz, and F. Jug, “Noise2Void - Learning Denoising From Single Noisy Images,” in 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, Jun. 2019, pp. 2124–2132. doi: [10.1109/CVPR.2019.00223](https://doi.org/10.1109/CVPR.2019.00223).
- [3] C. Broaddus, A. Krull, M. Weigert, U. Schmidt, and G. Myers, “Removing Structured Noise with Self-Supervised Blind-Spot Networks,” in 2020 IEEE 17th International Symposium on Biomedical Imaging (ISBI), IEEE, Apr. 2020, pp. 159–163. doi: [10.1109/ISBI45749.2020.9098336](https://doi.org/10.1109/ISBI45749.2020.9098336).
- [4] V. Lempitsky, A. Vedaldi, and D. Ulyanov, “Deep Image Prior,” in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, IEEE, Jun. 2018, pp. 9446–9454. doi: [10.1109/CVPR.2018.00984](https://doi.org/10.1109/CVPR.2018.00984).
- [5] A. A. Hendriksen, D. M. Pelt, and K. J. Batenburg, "Noise2Inverse: Self-Supervised Deep Convolutional Denoising for Tomography," IEEE Transactions on Computational Imaging, vol. 6, pp. 1320–1335, 2020, doi: [10.1109/TCI.2020.3019647](https://doi.org/10.1109/TCI.2020.3019647).

## Getting Started

It takes just a few steps to setup Auto-Denoise on your machine.

### Installing with conda

We recommend using [Miniforge](https://github.com/conda-forge/miniforge).
Once installed `miniforge`, simply install `autoden` with:
```bash
conda install auto-denoise -c n-vigano
```

### Installing from PyPI

Simply install with:
```bash
python -m pip install auto-denoise
```

If you are on jupyter, and don't have the rights to install packages system-wide, then you can install with:
```python
! python -m pip install --user auto-denoise
```

### Installing from source

To install Auto-Denoise, simply clone this github.com project with either:
```bash
git clone https://github.com/CEA-MetroCarac/auto-denoise.git auto-denoise
```
or:
```bash
git clone git@github.com:CEA-MetroCarac/auto-denoise.git auto-denoise
```

Then go to the cloned directory and run `pip` installer:
```bash
cd auto-denoise
pip install -e .
```

## How to contribute

Contributions are always welcome. Please submit pull requests against the `main` branch.

If you have any issues, questions, or remarks, then please open an issue on github.com.
