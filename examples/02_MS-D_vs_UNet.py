"""
This example shows a comparison of supervised and self-supervised denoising.

@author: Nicola VIGANÒ, CEA-MEM, Grenoble, France
"""

import matplotlib.pyplot as plt
import numpy as np
import skimage.color as skc
import skimage.data as skd
import skimage.transform as skt
from numpy.typing import NDArray
from tqdm.auto import tqdm
import autoden as ad


USE_CAMERA_MAN = True
NUM_IMGS_TRN = 4
NUM_IMGS_TST = 2
NUM_IMGS_TOT = NUM_IMGS_TRN + NUM_IMGS_TST

EPOCHS = 1024 * 2
REG_TV_VAL = 1e-7

if USE_CAMERA_MAN:
    img_orig = skd.camera()
    img_orig = skt.downscale_local_mean(img_orig, 4)
else:
    img_orig = skd.cat()
    img_orig = skc.rgb2gray(img_orig)
    img_orig *= 255 / img_orig.max()

imgs_noisy: NDArray = np.stack(
    [
        (img_orig + 20 * np.random.randn(*img_orig.shape))  # .clip(0, 255)
        for _ in tqdm(range(NUM_IMGS_TOT), desc="Create noisy images")
    ],
    axis=0,
)
tst_inds = np.arange(NUM_IMGS_TRN, NUM_IMGS_TOT)

print(f"Img orig -> [{img_orig.min()}, {img_orig.max()}], Img noisy -> [{imgs_noisy[0].min()}, {imgs_noisy[0].max()}]")
print(f"Img shape: {img_orig.shape}")

# fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
# axs[0].imshow(img_orig)
# axs[1].imshow(imgs_noisy[0])
# fig.tight_layout()
# plt.show(block=False)

net_params_unet = ad.NetworkParamsUNet(n_features=16)
net_params_msd_dil = ad.NetworkParamsMSD(n_features=1, n_layers=16, use_dilations=True)
net_params_msd_samp = ad.NetworkParamsMSD(n_features=1, n_layers=16, use_dilations=False)

denoiser_un = ad.N2N(model=net_params_unet, reg_val=REG_TV_VAL)
n2n_data = denoiser_un.prepare_data(imgs_noisy)
denoiser_un.train(*n2n_data, epochs=EPOCHS)

denoiser_md = ad.N2N(model=net_params_msd_dil, reg_val=REG_TV_VAL)
denoiser_md.train(*n2n_data, epochs=EPOCHS)

denoiser_ms = ad.N2N(model=net_params_msd_samp, reg_val=REG_TV_VAL)
denoiser_ms.train(*n2n_data, epochs=EPOCHS)

den_un = denoiser_un.infer(n2n_data[0])
den_md = denoiser_md.infer(n2n_data[0])
den_ms = denoiser_ms.infer(n2n_data[0])

fig, axs = plt.subplots(2, 3, sharex=True, sharey=True)
axs[0, 0].imshow(img_orig)
axs[0, 0].set_title("Original image")
axs[0, 1].imshow(imgs_noisy[0])
axs[0, 1].set_title("Noisy image")
# axs[0, 2].imshow(den_sup)
# axs[0, 2].set_title("Denoised supervised")
axs[1, 0].imshow(den_un)
axs[1, 0].set_title("Denoised UNet")
axs[1, 1].imshow(den_md)
axs[1, 1].set_title("Denoised MS-D dil")
axs[1, 2].imshow(den_ms)
axs[1, 2].set_title("Denoised MS-D samp")
fig.tight_layout()
plt.show(block=False)

from corrct.processing.post import plot_frcs

plot_frcs([(img, img_orig) for img in [den_un, den_md, den_ms]], ["UNet", "MS-D dil", "MS-D samp"])
