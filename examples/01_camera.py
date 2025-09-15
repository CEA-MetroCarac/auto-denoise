"""
This example shows a comparison of supervised and self-supervised denoising.

@author: Nicola VIGANÒ, UGA, CEA-IRIG, MEM, Grenoble, France
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

EPOCHS = 1024
REG_TV_VAL = 1e-7

if USE_CAMERA_MAN:
    img_orig = skd.camera()
    img_orig = skt.downscale_local_mean(img_orig, 4)
else:
    img_orig = skd.cat()
    img_orig = skc.rgb2gray(img_orig)
    img_orig *= 255 / img_orig.max()

imgs_noisy: NDArray = np.stack(
    [(img_orig + 20 * np.random.randn(*img_orig.shape)) for _ in tqdm(range(NUM_IMGS_TOT), desc="Create noisy images")],
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

net_params = ad.NetworkParamsUNet(n_features=16)

denoiser_sup = ad.Supervised(model=net_params, reg_val=REG_TV_VAL)
sup_data = denoiser_sup.prepare_data(imgs_noisy, img_orig, num_tst_ratio=NUM_IMGS_TST / NUM_IMGS_TOT)
denoiser_sup.train(*sup_data, epochs=EPOCHS)

denoiser_n2v = ad.N2V(model=net_params, reg_val=REG_TV_VAL)
denoiser_n2v.train(imgs_noisy, epochs=EPOCHS, tst_inds=tst_inds)

denoiser_n2n = ad.N2N(model=net_params, reg_val=REG_TV_VAL)
n2n_data = denoiser_n2n.prepare_data(imgs_noisy)
denoiser_n2n.train(*n2n_data, epochs=EPOCHS)

denoiser_dip = ad.DIP(model=net_params, reg_val=REG_TV_VAL)
dip_data = denoiser_dip.prepare_data(imgs_noisy)
denoiser_dip.train(*dip_data, epochs=EPOCHS)

den_sup = denoiser_sup.infer(sup_data[0]).mean(0)
den_n2v = denoiser_n2v.infer(imgs_noisy).mean(0)
den_n2n = denoiser_n2n.infer(n2n_data[0])
den_dip = denoiser_dip.infer(dip_data[0])

fig, axs = plt.subplots(2, 3, sharex=True, sharey=True)
axs[0, 0].imshow(img_orig)
axs[0, 0].set_title("Original image")
axs[0, 1].imshow(imgs_noisy[0])
axs[0, 1].set_title("Noisy image")
axs[0, 2].imshow(den_sup)
axs[0, 2].set_title("Denoised supervised")
axs[1, 0].imshow(den_n2v)
axs[1, 0].set_title("Denoised N2V")
axs[1, 1].imshow(den_n2n)
axs[1, 1].set_title("Denoised N2N")
axs[1, 2].imshow(den_dip)
axs[1, 2].set_title("Denoised DIP")
fig.tight_layout()
plt.show(block=False)
