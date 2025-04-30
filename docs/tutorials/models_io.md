# Loading and storing models

Auto-denoise offers the ability to store models for later use. The reasons why one would want to do that include: reproducibility for batches of tests, pre-training, fine-tuning, etc.
While Auto-denoise provides a handful of pre-implemented model types, you can always use your own model. In order to use your customized models in the provided algorithms, they need to inherit from `PyTorch`'s `nn.Module` class.
At the end of this tutorial, we will explain how to make your own models compatible with the storing/loading mechanics of auto-denoise.

## Creating models

Auto-denoise offers two ways to create a pre-configured model: (a) model configuration class, and (b) instantiating the model directly:
```python title="(a) Configuration class"
import autoden as ad

model_def = ad.models.config.NetworkParamsUNet(n_features=32, n_levels=3)
model = model_def.get_model()
```
or simply:
```python title="(b) Direct model instantiation"
import autoden as ad

model = ad.models.unet.UNet(1, 1, n_features=32, n_levels=3)
```

While the second is more compact, it might be useful to use a model configuration class when we just want to pass around a description of the model architecture, without having to instantiate and initialize its weights.

## Storing models

Storing models to file can be done through the following function:
```python 
ad.models.io.save_model("file_dest.pt", model)
```
Optionally, `save_model` can store the optimizer state and epoch number.
The model weights, name, and architecture are all saved in the same file.

## Loading models

Stored models can be loaded with `PyTorch`'s `load` function, and then the `create_network` function from auto-denoise:
```python 
from torch import load as load_model

model_dict = load_model("file_dest.pt")
model = ad.models.config.create_network(model_dict)
```

## Making your model compatible

To make your custom model compatible with `auto-denoise`'s storing/loading mechanics, you need to implement the `SerializableModel` protocol from `ad.models.config`:

```python title="SerializableModel"
from collections.abc import Mapping
from typing import Protocol, runtime_checkable


@runtime_checkable
class SerializableModel(Protocol):
    """
    Protocol for serializable models.

    Provides a dictionary containing the initialization parameters of the model.
    """

    init_params: Mapping
```

The `init_params` dictionary should contain the input arguments necessary to initialize your model.
As an instructive example, the following is the implementation from the model `UNet`:

```python title="UNet initialization parameters storing"
class UNet(nn.Module):
    """U-net model."""

    def __init__(
        self,
        n_channels_in: int,
        n_channels_out: int,
        n_features: int = 32,
        n_levels: int = 3,
        n_channels_skip: int | None = None,
        bilinear: bool = True,
        pad_mode: str = "replicate",
        device: str = "cuda" if pt.cuda.is_available() else "cpu",
        verbose: bool = False,
    ):
        init_params = locals()
        del init_params["self"]
        del init_params["__class__"]

        super().__init__()
        self.init_params = init_params
        ...
```

Currently, the `create_network` function only knows the pre-configured models, so for the time being you will have to patch that function too.
In the future we will provide mechanics to register your model types.