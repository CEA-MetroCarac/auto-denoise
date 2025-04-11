# Auto-Denoise

[![ci](https://github.com/CEA-MetroCarac/auto-denoise/workflows/ci/badge.svg)](https://github.com/CEA-MetroCarac/auto-denoise/actions?query=workflow%3Aci)
[![documentation](https://img.shields.io/badge/docs-mkdocs%20material-blue.svg?style=flat)](https://CEA-MetroCarac.github.io/auto-denoise/)
[![pypi version](https://img.shields.io/pypi/v/auto-denoise.svg)](https://pypi.org/project/auto-denoise/)
[![gitter](https://badges.gitter.im/join%20chat.svg)](https://gitter.im/auto-denoise/community)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Auto-denoise (autoden) provides implementations for a small selection of unsupervised and self-supervised CNN denoising methods.
These methods currently include:
* Noise2Noise (N2N) - A self-supervised denoising method using pairs of images of the same object [1].
* Noise2Void (N2V) - A self-supervised denoising method capable of working with a single image [2]. We have also implemented a later development of the method that can work with structured noise [3].
* Deep Image Prior (DIP) - An unsupervised denoising/upsampling/deconvolution method that can also work with a single image [4].

- [1] Lehtinen, J., Munkberg, J., Hasselgren, J., Laine, S., Karras, T., Aittala, M., & Aila, T. (2018). Noise2Noise: Learning Image Restoration without Clean Data. In J. Dy & A. Krause (Eds.), Proceedings of the 35th International Conference on Machine Learning (Vol. 80, pp. 2965–2974). PMLR. https://proceedings.mlr.press/v80/lehtinen18a.html
- [2] Krull, A., Buchholz, T.-O., & Jug, F. (2019). Noise2Void - Learning Denoising From Single Noisy Images. 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2124–2132. https://doi.org/10.1109/CVPR.2019.00223
- [3] Broaddus, C., Krull, A., Weigert, M., Schmidt, U., & Myers, G. (2020). Removing Structured Noise with Self-Supervised Blind-Spot Networks. 2020 IEEE 17th International Symposium on Biomedical Imaging (ISBI), 159–163. https://doi.org/10.1109/ISBI45749.2020.9098336
- [4] Lempitsky, V., Vedaldi, A., & Ulyanov, D. (2018). Deep Image Prior. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, 9446–9454. https://doi.org/10.1109/CVPR.2018.00984

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
