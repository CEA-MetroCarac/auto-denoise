"""
This example shows a comparison of supervised and self-supervised denoising.

@author: Nicola VIGANÒ, CEA-MEM, Grenoble, France
"""

import matplotlib.pyplot as plt
import numpy as np
import skimage.color as skc
import skimage.data as skd
from numpy.typing import NDArray
from tqdm.auto import tqdm
import autoden as ad


USE_CAMERA_MAN = True

EPOCHS = 2048
REG_TV_VAL = None

if USE_CAMERA_MAN:
    img_orig = skd.camera()
else:
    img_orig = skd.cat()
    img_orig = skc.rgb2gray(img_orig)
    img_orig *= 255 / img_orig.max()

imgs_noisy: NDArray = np.stack(
    [(img_orig + 50 * np.random.randn(*img_orig.shape)).clip(0, 255) for _ in tqdm(range(12), desc="Create noisy images")],
    axis=0,
)
dset_split = ad.DatasetSplit.create_sequential(num_trn_imgs=8, num_tst_imgs=4)

print(f"Img orig -> [{img_orig.min()}, {img_orig.max()}], Img noisy -> [{imgs_noisy[0].min()}, {imgs_noisy[0].max()}]")
print(f"Img shape: {img_orig.shape}")

# fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
# axs[0].imshow(img_orig)
# axs[1].imshow(imgs_noisy[0])
# fig.tight_layout()
# plt.show(block=False)

net_params = ad.NetworkParamsUNet(n_features=16)

denoiser_sup = ad.N2N(network_type=net_params, reg_tv_val=REG_TV_VAL)
denoiser_sup.train_supervised(imgs_noisy, img_orig, epochs=EPOCHS, dset_split=dset_split)

denoiser_n2n = ad.N2N(network_type=net_params, reg_tv_val=REG_TV_VAL)
denoiser_n2n.train_selfsupervised(imgs_noisy, epochs=EPOCHS)

denoiser_n2v = ad.N2V(network_type=net_params, reg_tv_val=REG_TV_VAL)
denoiser_n2v.train_selfsupervised(imgs_noisy, epochs=EPOCHS, dset_split=dset_split)

denoiser_dip = ad.DIP(network_type=net_params, reg_tv_val=REG_TV_VAL)
inp_dip = denoiser_dip.train_unsupervised(imgs_noisy, epochs=EPOCHS)

den_sup = denoiser_sup.infer(imgs_noisy).mean(0)
den_n2n = denoiser_n2n.infer(imgs_noisy).mean(0)
den_n2v = denoiser_n2v.infer(imgs_noisy).mean(0)
den_dip = denoiser_dip.infer(inp_dip).mean(0)

fig, axs = plt.subplots(2, 3, sharex=True, sharey=True)
axs[0, 0].imshow(img_orig)
axs[0, 0].set_title("Original image")
axs[0, 1].imshow(imgs_noisy[0])
axs[0, 1].set_title("Noisy image")
axs[0, 2].imshow(den_sup)
axs[0, 2].set_title("Denoised supervised")
axs[1, 0].imshow(den_n2n)
axs[1, 0].set_title("Denoised N2N")
axs[1, 1].imshow(den_n2v)
axs[1, 1].set_title("Denoised N2V")
axs[1, 2].imshow(den_dip)
axs[1, 2].set_title("Denoised DIP")
fig.tight_layout()
plt.show(block=False)
