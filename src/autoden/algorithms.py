"""
Implementation of various unsupervised and self-supervised denoising methods.
"""

import copy as cp
from collections.abc import Mapping, Sequence
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch as pt
from numpy.typing import NDArray
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from autoden import datasets
from autoden import losses
from autoden.models.config import NetworkParams, NetworkParamsDnCNN, NetworkParamsMSD, NetworkParamsUNet


def _get_normalization(vol: NDArray, percentile: float | None = None) -> tuple[float, float, float]:
    if percentile is not None:
        vol_sort = np.sort(vol.flatten())
        ind_min = int(np.fmax(vol_sort.size * percentile, 0))
        ind_max = int(np.fmin(vol_sort.size * (1 - percentile), vol_sort.size - 1))
        return vol_sort[ind_min], vol_sort[ind_max], vol_sort[ind_min : ind_max + 1].mean()
    else:
        return vol.min(), vol.max(), vol.mean()


def _random_probe_mask(
    img_shape: Sequence[int] | NDArray,
    mask_shape: int | Sequence[int] | NDArray = [1, 5],
    ratio_blind_spots: float = 0.02,
    verbose: bool = False,
) -> NDArray:
    img_shape = np.array(img_shape, dtype=int)

    if isinstance(mask_shape, int) or len(mask_shape) == 1:
        mask_shape = np.ones_like(img_shape) * mask_shape
    elif len(img_shape) != len(mask_shape):
        raise ValueError(
            f"Input mask (ndim: {len(mask_shape)}) should have the same dimensionality as the image (ndim: {img_shape.ndim})"
        )
    mask_shape = np.array(mask_shape, dtype=int)

    mask = np.zeros(img_shape, dtype=np.uint8)
    num_blind_spots = int(mask.size * ratio_blind_spots)
    bspot_coords = [np.random.randint(0, edge, num_blind_spots) for edge in img_shape]

    mask_hlf_size = np.array(mask_shape) // 2
    mask_pix_inds = [
        np.linspace(-dim_h_size, dim_h_size, dim_size, dtype=int) for dim_h_size, dim_size in zip(mask_hlf_size, mask_shape)
    ]
    mask_pix_inds = np.meshgrid(*mask_pix_inds, indexing="ij")
    mask_pix_inds = np.stack(mask_pix_inds, axis=-1).reshape([-1, len(img_shape)])

    for mask_pix_coords in mask_pix_inds:
        valid = [
            np.logical_and((bspot_coords[ii] + coord) >= 0, (bspot_coords[ii] + coord) < img_shape[ii])
            for ii, coord in enumerate(mask_pix_coords)
        ]
        valid = np.all(valid, axis=0)

        valid_inds = [dim_inds[valid] + ii for dim_inds, ii in zip(bspot_coords, mask_pix_coords)]
        mask[tuple(valid_inds)] = 1

    mask[tuple(bspot_coords)] = 2

    if verbose:
        fig, axs = plt.subplots(1, 1)
        axs.imshow(mask)
        fig.tight_layout()

    return mask


def _create_network(
    network: str | NetworkParams,
    device: str = "cuda" if pt.cuda.is_available() else "cpu",
) -> pt.nn.Module:
    if isinstance(network, str):
        if network.lower() == "msd":
            network = NetworkParamsMSD()
        elif network.lower() == "unet":
            network = NetworkParamsUNet()
        elif network.lower() == "dncnn":
            network = NetworkParamsDnCNN()
        else:
            raise ValueError(f"Invalid network name: {network}")

    net = network.get_model(device)

    print(f"Model {net.__class__.__name__} - num. parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad)}")
    return net


def _create_optimizer(network: pt.nn.Module, algo: str = "adam", learning_rate: float = 1e-3, weight_decay: float = 1e-2):
    if algo.lower() == "adam":
        return pt.optim.AdamW(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif algo.lower() == "rmsprop":
        return pt.optim.RMSprop(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")


class DatasetSplit:
    """Store the dataset split indices, between training and validation."""

    trn_inds: NDArray[np.integer]
    tst_inds: NDArray[np.integer] | None

    def __init__(self, trn_inds: NDArray, tst_inds: NDArray | None = None) -> None:
        self.trn_inds = np.array(trn_inds)
        self.tst_inds = np.array(tst_inds) if tst_inds is not None else None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(\n  Training indices: {self.trn_inds}\n  Testing indices: {self.tst_inds}\n)"

    @staticmethod
    def create_sequential(num_trn_imgs: int, num_tst_imgs: int | None = None) -> "DatasetSplit":
        return DatasetSplit(
            np.arange(num_trn_imgs), np.arange(num_trn_imgs, num_trn_imgs + num_tst_imgs) if num_tst_imgs is not None else None
        )

    @staticmethod
    def create_random(num_trn_imgs: int, num_tst_imgs: int | None, tot_num_imgs: int | None = None) -> "DatasetSplit":
        if tot_num_imgs is None:
            tot_num_imgs = num_trn_imgs + num_tst_imgs if num_tst_imgs is not None else 0
        inds = np.arange(tot_num_imgs)
        inds = np.random.permutation(inds)
        return DatasetSplit(
            inds[:num_trn_imgs], inds[num_trn_imgs : num_trn_imgs + num_tst_imgs] if num_tst_imgs is not None else None
        )


class Denoiser:
    """Denoising images."""

    dataset_name: str
    n_channels: int

    data_scaling_inp: float | NDArray
    data_scaling_tgt: float | NDArray

    data_bias_inp: float | NDArray
    data_bias_tgt: float | NDArray

    net: pt.nn.Module

    device: str
    save_epochs: bool

    verbose: bool

    def __init__(
        self,
        dataset_name: str,
        network_type: str | NetworkParams,
        network_state: Mapping | None = None,
        data_scaling_inp: float | None = None,
        data_scaling_tgt: float | None = None,
        reg_tv_val: float | None = 1e-5,
        batch_size: int = 8,
        device: str = "cuda" if pt.cuda.is_available() else "cpu",
        save_epochs: bool = True,
        verbose: bool = True,
    ) -> None:
        """Initialize the noise2noise method.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset.
        network_type : Union[str, NetworkParams]
            Type of neural network to use
        network_state : Union[Mapping, None], optional
            Specific network state to load, by default None
        data_scaling_inp : Union[float, None], optional
            Scaling of the input data, by default None
        data_scaling_tgt : Union[float, None], optional
            Scaling of the output, by default None
        reg_tv_val : Union[float, None], optional
            Deep-image prior regularization value, by default 1e-5
        batch_size : int, optional
            Size of the batch, by default 8
        device : str, optional
            Device to use, by default "cuda" if cuda is available, otherwise "cpu"
        save_epochs : bool, optional
            Whether to save network states at each epoch, by default True
        verbose : bool, optional
            Whether to produce verbose output, by default True
        """
        self.dataset_name = dataset_name

        if isinstance(network_type, str):
            self.n_channels = 1
        else:
            self.n_channels = network_type.n_channels_in

        self.net = _create_network(network_type, device=device)

        if network_state is not None:
            if isinstance(network_state, int):
                self._load_state(network_state)
            else:
                self.net.load_state_dict(network_state)

        if data_scaling_inp is not None:
            self.data_scaling_inp = data_scaling_inp
        else:
            self.data_scaling_inp = 1
        if data_scaling_tgt is not None:
            self.data_scaling_tgt = data_scaling_tgt
        else:
            self.data_scaling_tgt = 1

        self.data_bias_inp = 0
        self.data_bias_tgt = 0

        self.reg_val = reg_tv_val
        self.batch_size = batch_size
        self.device = device
        self.save_epochs = save_epochs
        self.verbose = verbose

    def train_supervised(self, inp: NDArray, tgt: NDArray, epochs: int, dset_split: DatasetSplit, algo: str = "adam"):
        """Supervised training.

        Parameters
        ----------
        inp : NDArray
            The input images
        tgt : NDArray
            The target images
        epochs : int
            Number of training epochs
        dset_split : DatasetSplit
            How to split the dataset in training and validation set
        algo : str, optional
            Learning algorithm to use, by default "adam"
        """
        if tgt.ndim == (inp.ndim - 1):
            tgt = np.tile(tgt[None, ...], [inp.shape[0], *np.ones_like(tgt.shape)])

        range_vals_inp = _get_normalization(inp, percentile=0.001)
        range_vals_tgt = _get_normalization(tgt, percentile=0.001)

        self.data_scaling_inp = 1 / (range_vals_inp[1] - range_vals_inp[0])
        self.data_scaling_tgt = 1 / (range_vals_tgt[1] - range_vals_tgt[0])

        self.data_bias_inp = inp.mean() * self.data_scaling_inp
        self.data_bias_tgt = tgt.mean() * self.data_scaling_tgt

        # Rescale the datasets
        inp = inp * self.data_scaling_inp - self.data_bias_inp
        tgt = tgt * self.data_scaling_tgt - self.data_bias_tgt

        # Create datasets
        dset_trn = datasets.SupervisedDataset(inp[dset_split.trn_inds], tgt[dset_split.trn_inds], device=self.device)
        dset_tst = datasets.SupervisedDataset(inp[dset_split.tst_inds], tgt[dset_split.tst_inds], device=self.device)

        dl_trn = DataLoader(dset_trn, batch_size=self.batch_size)
        dl_tst = DataLoader(dset_tst, batch_size=self.batch_size * 16)

        reg = losses.LossTGV(self.reg_val, reduction="mean") if self.reg_val is not None else None
        loss_trn, loss_tst = self._train_selfsimilar_big(dl_trn, dl_tst, epochs=epochs, algo=algo, regularizer=reg)

        if self.verbose:
            self._plot_loss_curves(loss_trn, loss_tst, f"Supervised {algo.upper()}")

    def _train_selfsimilar_big(
        self,
        dl_trn: DataLoader,
        dl_tst: DataLoader,
        epochs: int,
        algo: str = "adam",
        regularizer: losses.LossRegularizer | None = None,
    ) -> tuple[NDArray, NDArray]:
        losses_trn = []
        losses_tst = []
        loss_data_fn = pt.nn.MSELoss(reduction="sum")
        optim = _create_optimizer(self.net, algo=algo)

        best_epoch = -1
        best_loss_tst = +np.inf
        best_state = self.net.state_dict()
        best_optim = optim.state_dict()

        dset_trn_size = len(dl_trn)
        dset_tst_size = len(dl_tst)

        for epoch in tqdm(range(epochs), desc=f"Training {algo.upper()}"):
            # Train
            self.net.train()
            loss_trn_val = 0
            for inp_trn, tgt_trn in dl_trn:
                # inp_trn = inp_trn.to(self.device, non_blocking=True)
                # tgt_trn = tgt_trn.to(self.device, non_blocking=True)

                optim.zero_grad()
                out_trn = self.net(inp_trn)
                loss_trn = loss_data_fn(out_trn, tgt_trn)
                if regularizer is not None:
                    loss_trn += regularizer(out_trn)
                loss_trn.backward()

                loss_trn_val += loss_trn.item()

                optim.step()

            losses_trn.append(loss_trn_val / dset_trn_size)

            # Test
            self.net.eval()
            loss_tst_val = 0
            with pt.inference_mode():
                for inp_tst, tgt_tst in dl_tst:
                    # inp_tst = inp_tst.to(self.device, non_blocking=True)
                    # tgt_tst = tgt_tst.to(self.device, non_blocking=True)

                    out_tst = self.net(inp_tst)
                    loss_tst = loss_data_fn(out_tst, tgt_tst)

                    loss_tst_val += loss_tst.item()

                losses_tst.append(loss_tst_val / dset_tst_size)

            # Check improvement
            if losses_tst[-1] < best_loss_tst if losses_tst[-1] is not None else False:
                best_loss_tst = losses_tst[-1]
                best_epoch = epoch
                best_state = cp.deepcopy(self.net.state_dict())
                best_optim = cp.deepcopy(optim.state_dict())

            # Save epoch
            if self.save_epochs:
                self._save_state(epoch, self.net.state_dict(), optim.state_dict())

        print(f"Best epoch: {best_epoch}, with tst_loss: {best_loss_tst:.5}")
        if self.save_epochs:
            self._save_state(best_epoch, best_state, best_optim, is_final=True)

        self.net.load_state_dict(best_state)

        return np.array(losses_trn), np.array(losses_tst)

    def _train_pixelmask_small(
        self,
        inp: NDArray,
        tgt: NDArray,
        mask_trn: NDArray,
        epochs: int,
        algo: str = "adam",
        regularizer: losses.LossRegularizer | None = None,
    ) -> tuple[NDArray, NDArray]:
        losses_trn = []
        losses_tst = []
        loss_data_fn = pt.nn.MSELoss(reduction="mean")
        optim = _create_optimizer(self.net, algo=algo)

        best_epoch = -1
        best_loss_tst = +np.inf
        best_state = self.net.state_dict()
        best_optim = optim.state_dict()

        n_dims = inp.ndim
        if n_dims == 2:
            inp = inp[None, None, ...]
        else:
            inp = inp[:, None, ...]

        inp_t = pt.tensor(inp, device=self.device)
        tgt_trn = pt.tensor(tgt[mask_trn], device=self.device)
        tgt_tst = pt.tensor(tgt[np.logical_not(mask_trn)], device=self.device)

        mask_trn_t = pt.tensor(mask_trn, device=self.device)
        mask_tst_t = pt.tensor(np.logical_not(mask_trn), device=self.device)

        self.net.train()
        for epoch in tqdm(range(epochs), desc=f"Training {algo.upper()}"):
            # Train
            optim.zero_grad()
            out_t: pt.Tensor = self.net(inp_t)
            if n_dims == 2:
                out_t_mask = out_t[0, 0]
            else:
                out_t_mask = out_t[:, 0]
            if tgt.ndim == 3 and out_t_mask.ndim == 2:
                out_t_mask = pt.tile(out_t_mask[None, :, :], [tgt.shape[-3], 1, 1])

            out_trn = out_t_mask[mask_trn_t]

            loss_trn = loss_data_fn(out_trn, tgt_trn)
            if regularizer is not None:
                loss_trn += regularizer(out_t)
            loss_trn.backward()

            losses_trn.append(loss_trn.item())
            optim.step()

            # Test
            out_tst = out_t_mask[mask_tst_t]
            loss_tst = loss_data_fn(out_tst, tgt_tst)
            losses_tst.append(loss_tst.item())

            # Check improvement
            if losses_tst[-1] < best_loss_tst if losses_tst[-1] is not None else False:
                best_loss_tst = losses_tst[-1]
                best_epoch = epoch
                best_state = cp.deepcopy(self.net.state_dict())
                best_optim = cp.deepcopy(optim.state_dict())

            # Save epoch
            if self.save_epochs:
                self._save_state(epoch, self.net.state_dict(), optim.state_dict())

        print(f"Best epoch: {best_epoch}, with tst_loss: {best_loss_tst:.5}")
        if self.save_epochs:
            self._save_state(best_epoch, best_state, best_optim, is_final=True)

        self.net.load_state_dict(best_state)

        losses_trn = np.array(losses_trn)
        losses_tst = np.array(losses_tst)

        return losses_trn, losses_tst

    def _save_state(self, epoch_num: int, net_state: Mapping, optim_state: Mapping, is_final: bool = False) -> None:
        epochs_base_path = Path(self.dataset_name) / "weights"
        epochs_base_path.mkdir(parents=True, exist_ok=True)

        if is_final:
            pt.save({"epoch": epoch_num, "state_dict": net_state, "optimizer": optim_state}, epochs_base_path / "weights.pt")
        else:
            pt.save(
                {"epoch": epoch_num, "state_dict": net_state, "optimizer": optim_state},
                epochs_base_path / f"weights_epoch_{epoch_num}.pt",
            )

    def _load_state(self, epoch_num: int | None = None) -> None:
        epochs_base_path = Path(self.dataset_name) / "weights"
        if not epochs_base_path.exists():
            raise ValueError("No state to load!")

        if epoch_num is None or epoch_num == -1:
            state_path = epochs_base_path / "weights.pt"
        else:
            state_path = epochs_base_path / f"weights_epoch_{epoch_num}.pt"
        print(f"Loading state path: {state_path}")
        state_dict = pt.load(state_path)
        self.net.load_state_dict(state_dict["state_dict"])

    def _plot_loss_curves(self, train_loss: NDArray, test_loss: NDArray, title: str | None = None) -> None:
        test_argmin = int(np.argmin(test_loss))
        fig, axs = plt.subplots(1, 1, figsize=[7, 2.6])
        if title is not None:
            axs.set_title(title)
        axs.semilogy(np.arange(train_loss.size), train_loss, label="training loss")
        axs.semilogy(np.arange(test_loss.size) + 1, test_loss, label="test loss")
        axs.stem(test_argmin + 1, test_loss[test_argmin], linefmt="C1--", markerfmt="C1o", label=f"Best epoch: {test_argmin}")
        axs.legend()
        axs.grid()
        fig.tight_layout()
        plt.show(block=False)

    def infer(self, inp: NDArray) -> NDArray:
        """Inference, given an initial stack of images.

        Parameters
        ----------
        inp : NDArray
            The input stack of images

        Returns
        -------
        NDArray
            The denoised stack of images
        """
        # Rescale input
        inp = inp * self.data_scaling_inp - self.data_bias_inp

        # Create datasets
        dset = datasets.InferenceDataset(inp, device=self.device)

        dtl = DataLoader(dset, batch_size=self.batch_size)

        output = self._infer(dtl)

        # Rescale output
        return (output + self.data_bias_tgt) / self.data_scaling_tgt

    def _infer(self, dtl: DataLoader) -> NDArray:
        self.net.eval()
        output = []
        with pt.inference_mode():
            for inp in tqdm(dtl, desc="Inference"):
                inp = inp.to(self.device, non_blocking=True)

                out = self.net(inp)
                output.append(out.cpu().numpy())

        output = np.concatenate(output, axis=0)
        if output.shape[1] == 1:
            output = np.squeeze(output, axis=1)
        return output


class N2N(Denoiser):
    """Self-supervised denoising from pairs of images."""

    def train_selfsupervised(
        self, inp: NDArray, epochs: int, num_tst_ratio: float = 0.2, strategy: str = "1:X", algo: str = "adam"
    ) -> None:
        range_vals_tgt = range_vals_inp = _get_normalization(inp, percentile=0.001)

        self.data_scaling_inp = 1 / (range_vals_inp[1] - range_vals_inp[0])
        self.data_scaling_tgt = 1 / (range_vals_tgt[1] - range_vals_tgt[0])

        self.data_bias_inp = inp.mean() * self.data_scaling_inp
        self.data_bias_tgt = inp.mean() * self.data_scaling_tgt

        # Rescale the datasets
        inp = inp * self.data_scaling_inp - self.data_bias_inp

        mask_trn = np.ones_like(inp, dtype=bool)
        rnd_inds = np.random.random_integers(low=0, high=mask_trn.size - 1, size=int(mask_trn.size * num_tst_ratio))
        mask_trn[np.unravel_index(rnd_inds, shape=mask_trn.shape)] = False

        inp_x = np.stack([np.delete(inp, obj=ii, axis=0).mean(axis=0) for ii in range(len(inp))], axis=0)
        if strategy.upper() == "1:X":
            tmp_inp = inp
            tmp_tgt = inp_x
        elif strategy.upper() == "X:1":
            tmp_inp = inp_x
            tmp_tgt = inp
        else:
            raise ValueError(f"Strategy {strategy} not implemented. Please choose one of: ['1:X', 'X:1']")

        tmp_inp = tmp_inp.astype(np.float32)
        tmp_tgt = tmp_tgt.astype(np.float32)

        reg = losses.LossTGV(self.reg_val, reduction="mean") if self.reg_val is not None else None
        losses_trn, losses_tst = self._train_pixelmask_small(
            tmp_inp, tmp_tgt, mask_trn, epochs=epochs, algo=algo, regularizer=reg
        )

        if self.verbose:
            self._plot_loss_curves(losses_trn, losses_tst, f"Self-supervised {self.__class__.__name__} {algo.upper()}")


class N2V(Denoiser):
    "Self-supervised denoising from single images."

    # def train_selfsupervised(
    #     self,
    #     inp: NDArray,
    #     epochs: int,
    #     dset_split: DatasetSplit,
    #     mask_shape: int | Sequence[int] | NDArray = 1,
    #     ratio_blind_spot: float = 0.015,
    #     algo: str = "adam",
    # ):
    #     """Self-supervised training.

    #     Parameters
    #     ----------
    #     inp : NDArray
    #         The input images, which will also be targets
    #     epochs : int
    #         Number of training epochs
    #     dset_split : DatasetSplit
    #         How to split the dataset in training and validation set
    #     mask_shape : int | Sequence[int] | NDArray
    #         Shape of the blind spot mask, by default 1.
    #     algo : str, optional
    #         Learning algorithm to use, by default "adam"
    #     """
    #     range_vals_tgt = range_vals_inp = _get_normalization(inp, percentile=0.001)

    #     self.data_scaling_inp = 1 / (range_vals_inp[1] - range_vals_inp[0])
    #     self.data_scaling_tgt = 1 / (range_vals_tgt[1] - range_vals_tgt[0])

    #     self.data_bias_inp = inp.mean() * self.data_scaling_inp
    #     self.data_bias_tgt = inp.mean() * self.data_scaling_tgt

    #     # Rescale the datasets
    #     inp = inp * self.data_scaling_inp - self.data_bias_inp

    #     inp_trn = inp[dset_split.trn_inds]
    #     inp_tst = inp[dset_split.tst_inds]

    #     dsets_trn = datasets.NumpyDataset(inp_trn, n_channels=self.n_channels)
    #     dsets_tst = datasets.NumpyDataset(inp_tst, n_channels=self.n_channels)

    #     # Create datasets
    #     dset_trn = datasets.InferenceDataset(dsets_trn, device=self.device)
    #     dset_tst = datasets.InferenceDataset(dsets_tst, device=self.device)

    #     dl_trn = DataLoader(dset_trn, batch_size=self.batch_size)
    #     dl_tst = DataLoader(dset_tst, batch_size=self.batch_size * 16)

    #     reg = losses.LossTGV(self.reg_val, reduction="mean") if self.reg_val is not None else None
    #     losses_trn, losses_tst = self._train_n2v_selfsimilar_big(
    #         dl_trn, dl_tst, epochs=epochs, mask_shape=mask_shape, ratio_blind_spot=ratio_blind_spot, algo=algo, regularizer=reg
    #     )

    #     self._plot_loss_curves(losses_trn, losses_tst, f"Self-supervised {self.__class__.__name__} {algo.upper()}")

    # def _train_n2v_selfsimilar_big(
    #     self,
    #     dl_trn: DataLoader,
    #     dl_tst: DataLoader,
    #     epochs: int,
    #     mask_shape: int | Sequence[int] | NDArray,
    #     ratio_blind_spot: float,
    #     algo: str = "adam",
    #     regularizer: losses.LossRegularizer | None = None,
    # ) -> tuple[NDArray, NDArray]:
    #     losses_trn = []
    #     losses_tst = []
    #     loss_data_fn = pt.nn.MSELoss(reduction="sum")
    #     optim = _create_optimizer(self.net, algo=algo)

    #     best_epoch = -1
    #     best_loss_tst = +np.inf
    #     best_state = self.net.state_dict()
    #     best_optim = optim.state_dict()

    #     dset_trn_size = len(dl_trn)
    #     dset_tst_size = len(dl_tst)

    #     for epoch in tqdm(range(epochs), desc=f"Training {algo.upper()}"):
    #         # Train
    #         self.net.train()
    #         loss_trn_val = 0
    #         for inp_trn in dl_trn:
    #             inp_trn = pt.squeeze(inp_trn, dim=0).swapaxes(0, 1)
    #             mask = _random_probe_mask(inp_trn.shape[-2:], mask_shape, ratio_blind_spots=ratio_blind_spot)
    #             to_damage = np.where(mask > 0)
    #             to_check = np.where(mask > 1)
    #             inp_trn_damaged = pt.clone(inp_trn)
    #             size_to_damage = inp_trn_damaged[:, :, to_damage[0], to_damage[1]].shape
    #             inp_trn_damaged[:, :, to_damage[0], to_damage[1]] = pt.randn(
    #                 size_to_damage, device=inp_trn.device, dtype=inp_trn.dtype
    #             )

    #             optim.zero_grad()
    #             out_trn = self.net(inp_trn_damaged)
    #             out_to_check = out_trn[:, :, to_check[0], to_check[1]].flatten()
    #             ref_to_check = inp_trn[:, :, to_check[0], to_check[1]].flatten()
    #             loss_trn = loss_data_fn(out_to_check, ref_to_check)
    #             if regularizer is not None:
    #                 loss_trn += regularizer(out_trn)
    #             loss_trn.backward()

    #             loss_trn_val += loss_trn.item()

    #             optim.step()

    #         losses_trn.append(loss_trn_val / dset_trn_size)

    #         # Test
    #         self.net.eval()
    #         loss_tst_val = 0
    #         with pt.inference_mode():
    #             for inp_tst in dl_tst:
    #                 inp_tst = pt.squeeze(inp_tst, dim=0).swapaxes(0, 1)
    #                 mask = _random_probe_mask(inp_tst.shape[-2:], mask_shape, ratio_blind_spots=ratio_blind_spot)
    #                 to_damage = np.where(mask > 0)
    #                 to_check = np.where(mask > 1)
    #                 inp_tst_damaged = pt.clone(inp_tst)
    #                 size_to_damage = inp_tst_damaged[:, :, to_damage[0], to_damage[1]].shape
    #                 inp_tst_damaged[:, :, to_damage[0], to_damage[1]] = pt.randn(
    #                     size_to_damage, device=inp_tst.device, dtype=inp_tst.dtype
    #                 )

    #                 out_tst = self.net(inp_tst_damaged)
    #                 out_to_check = out_tst[:, :, to_check[0], to_check[1]].flatten()
    #                 ref_to_check = inp_tst[:, :, to_check[0], to_check[1]].flatten()
    #                 loss_tst = loss_data_fn(out_to_check, ref_to_check)

    #                 loss_tst_val += loss_tst.item()

    #             losses_tst.append(loss_tst_val / dset_tst_size)

    #         # Check improvement
    #         if losses_tst[-1] < best_loss_tst if losses_tst[-1] is not None else False:
    #             best_loss_tst = losses_tst[-1]
    #             best_epoch = epoch
    #             best_state = cp.deepcopy(self.net.state_dict())
    #             best_optim = cp.deepcopy(optim.state_dict())

    #         # Save epoch
    #         if self.save_epochs:
    #             self._save_state(epoch, self.net.state_dict(), optim.state_dict())

    #     print(f"Best epoch: {best_epoch}, with tst_loss: {best_loss_tst:.5}")
    #     if self.save_epochs:
    #         self._save_state(best_epoch, best_state, best_optim, is_final=True)

    #     self.net.load_state_dict(best_state)

    #     losses_trn = np.array(losses_trn)
    #     losses_tst = np.array(losses_tst)

    #     return losses_trn, losses_tst

    def train_selfsupervised(
        self,
        inp: NDArray,
        epochs: int,
        dset_split: DatasetSplit,
        mask_shape: int | Sequence[int] | NDArray = 1,
        ratio_blind_spot: float = 0.015,
        algo: str = "adam",
    ):
        """Self-supervised training.

        Parameters
        ----------
        inp : NDArray
            The input images, which will also be targets
        epochs : int
            Number of training epochs
        dset_split : DatasetSplit
            How to split the dataset in training and validation set
        mask_shape : int | Sequence[int] | NDArray
            Shape of the blind spot mask, by default 1.
        algo : str, optional
            Learning algorithm to use, by default "adam"
        """
        range_vals_tgt = range_vals_inp = _get_normalization(inp, percentile=0.001)

        self.data_scaling_inp = 1 / (range_vals_inp[1] - range_vals_inp[0])
        self.data_scaling_tgt = 1 / (range_vals_tgt[1] - range_vals_tgt[0])

        self.data_bias_inp = inp.mean() * self.data_scaling_inp
        self.data_bias_tgt = inp.mean() * self.data_scaling_tgt

        # Rescale the datasets
        inp = inp * self.data_scaling_inp - self.data_bias_inp

        inp_trn = inp[dset_split.trn_inds].astype(np.float32)
        inp_tst = inp[dset_split.tst_inds].astype(np.float32)

        reg = losses.LossTGV(self.reg_val, reduction="mean") if self.reg_val is not None else None
        losses_trn, losses_tst = self._train_n2v_pixelmask_small(
            inp_trn,
            inp_tst,
            epochs=epochs,
            mask_shape=mask_shape,
            ratio_blind_spot=ratio_blind_spot,
            algo=algo,
            regularizer=reg,
        )

        self._plot_loss_curves(losses_trn, losses_tst, f"Self-supervised {self.__class__.__name__} {algo.upper()}")

    def _train_n2v_pixelmask_small(
        self,
        inp_trn: NDArray,
        inp_tst: NDArray,
        epochs: int,
        mask_shape: int | Sequence[int] | NDArray,
        ratio_blind_spot: float,
        algo: str = "adam",
        regularizer: losses.LossRegularizer | None = None,
    ) -> tuple[NDArray, NDArray]:
        losses_trn = []
        losses_tst = []
        loss_data_fn = pt.nn.MSELoss(reduction="mean")
        optim = _create_optimizer(self.net, algo=algo)

        best_epoch = -1
        best_loss_tst = +np.inf
        best_state = self.net.state_dict()
        best_optim = optim.state_dict()

        inp_trn_t = pt.tensor(inp_trn, device=self.device)[:, None, ...]
        inp_tst_t = pt.tensor(inp_tst, device=self.device)[:, None, ...]

        for epoch in tqdm(range(epochs), desc=f"Training {algo.upper()}"):
            # Train
            self.net.train()

            mask = _random_probe_mask(inp_trn_t.shape[-2:], mask_shape, ratio_blind_spots=ratio_blind_spot)
            to_damage = np.where(mask > 0)
            to_check = np.where(mask > 1)
            inp_trn_damaged = pt.clone(inp_trn_t)
            size_to_damage = inp_trn_damaged[..., to_damage[0], to_damage[1]].shape
            inp_trn_damaged[..., to_damage[0], to_damage[1]] = pt.randn(
                size_to_damage, device=inp_trn_t.device, dtype=inp_trn_t.dtype
            )

            optim.zero_grad()
            out_trn = self.net(inp_trn_damaged)
            out_to_check = out_trn[..., to_check[0], to_check[1]].flatten()
            ref_to_check = inp_trn_t[..., to_check[0], to_check[1]].flatten()

            loss_trn = loss_data_fn(out_to_check, ref_to_check)
            if regularizer is not None:
                loss_trn += regularizer(out_trn)
            loss_trn.backward()

            losses_trn.append(loss_trn.item())
            optim.step()

            # Test
            self.net.eval()
            with pt.inference_mode():
                mask = _random_probe_mask(inp_tst_t.shape[-2:], mask_shape, ratio_blind_spots=ratio_blind_spot)
                to_damage = np.where(mask > 0)
                to_check = np.where(mask > 1)
                inp_tst_damaged = pt.clone(inp_tst_t)
                size_to_damage = inp_tst_damaged[..., to_damage[0], to_damage[1]].shape
                inp_tst_damaged[..., to_damage[0], to_damage[1]] = pt.randn(
                    size_to_damage, device=inp_tst_t.device, dtype=inp_tst_t.dtype
                )

                out_tst = self.net(inp_tst_damaged)
                out_to_check = out_tst[..., to_check[0], to_check[1]].flatten()
                ref_to_check = inp_tst_t[..., to_check[0], to_check[1]].flatten()
                loss_tst = loss_data_fn(out_to_check, ref_to_check)

                losses_tst.append(loss_tst.item())

            # Check improvement
            if losses_tst[-1] < best_loss_tst if losses_tst[-1] is not None else False:
                best_loss_tst = losses_tst[-1]
                best_epoch = epoch
                best_state = cp.deepcopy(self.net.state_dict())
                best_optim = cp.deepcopy(optim.state_dict())

            # Save epoch
            if self.save_epochs:
                self._save_state(epoch, self.net.state_dict(), optim.state_dict())

        print(f"Best epoch: {best_epoch}, with tst_loss: {best_loss_tst:.5}")
        if self.save_epochs:
            self._save_state(best_epoch, best_state, best_optim, is_final=True)

        self.net.load_state_dict(best_state)

        losses_trn = np.array(losses_trn)
        losses_tst = np.array(losses_tst)

        return losses_trn, losses_tst


class DIP(Denoiser):
    """Deep image prior."""

    def train_unsupervised(
        self, tgt: NDArray, epochs: int, inp: NDArray | None = None, num_tst_ratio: float = 0.2, algo: str = "adam"
    ) -> NDArray:
        if inp is None:
            tmp_inp = inp = np.random.normal(size=tgt.shape[-2:], scale=0.25).astype(tgt.dtype)
            self.data_scaling_inp = 1.0
            self.data_bias_inp = 0.0
        else:
            range_vals_inp = _get_normalization(inp, percentile=0.001)
            self.data_scaling_inp = 1 / (range_vals_inp[1] - range_vals_inp[0])
            self.data_bias_inp = range_vals_inp[2] * self.data_scaling_inp

            # Rescale input
            tmp_inp = inp * self.data_scaling_inp - self.data_bias_inp

        range_vals_tgt = _get_normalization(tgt, percentile=0.001)
        self.data_scaling_tgt = 1 / (range_vals_tgt[1] - range_vals_tgt[0])
        self.data_bias_tgt = range_vals_tgt[2] * self.data_scaling_tgt

        # Rescale target
        tmp_tgt = tgt * self.data_scaling_tgt - self.data_bias_tgt

        mask_trn = np.ones_like(tgt, dtype=bool)
        rnd_inds = np.random.random_integers(low=0, high=mask_trn.size - 1, size=int(mask_trn.size * num_tst_ratio))
        mask_trn[np.unravel_index(rnd_inds, shape=mask_trn.shape)] = False

        tmp_inp = tmp_inp.astype(np.float32)
        tmp_tgt = tmp_tgt.astype(np.float32)

        reg = losses.LossTGV(self.reg_val, reduction="mean") if self.reg_val is not None else None
        losses_trn, losses_tst = self._train_pixelmask_small(
            tmp_inp, tmp_tgt, mask_trn, epochs=epochs, algo=algo, regularizer=reg
        )

        if self.verbose:
            self._plot_loss_curves(losses_trn, losses_tst, f"Unsupervised {self.__class__.__name__} {algo.upper()}")

        return inp
