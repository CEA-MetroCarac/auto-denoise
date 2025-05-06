# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

<!-- insertion marker -->
## [v1.0.0](https://github.com/CEA-MetroCarac/auto-denoise/releases/tag/v1.0.0) - 2025-05-06

This is the first major release of Auto-denoise, providing initial implementations for a small selection of unsupervised and self-supervised CNN denoising methods. These methods currently include:

    Noise2Noise (N2N) - A self-supervised denoising method using pairs of images of the same object.
    Noise2Void (N2V) - A self-supervised denoising method capable of working with a single image. We have also implemented a later development of the method that can work with structured noise.
    Deep Image Prior (DIP) - An unsupervised denoising/upsampling/deconvolution method that can also work with a single image.
    Supervised denoising methods, and the tomography-specific Noise2Inverse (N2I) method.

We also provide a small set of pre-configured models for these algorithms: U-net, MS-D net, DnCNN, and a custom ResNet implementation.

<small>[Compare with first commit](https://github.com/CEA-MetroCarac/auto-denoise/compare/v0.0.0...v1.0.0)</small>
