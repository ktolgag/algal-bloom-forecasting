import os
import time
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchgeo.datasets import BoundingBox, RasterDataset  # type: ignore[attr-defined]
from torchgeo.samplers import GridGeoSampler, RandomGeoSampler  # type: ignore[attr-defined]
from torchgeo.samplers.constants import Units  # type: ignore[attr-defined]

plt.rcParams.update({"font.size": 8})


class RioNegroBiological(RasterDataset):
    filename_glob = "*_biological_100m.tif"
    filename_regex = r".*_(?P<date>\d{8})"
    date_format = "%Y%m%d"
    is_image = True
    separate_files = False
    all_bands = [
        "chlorophyll-a-100m",
        "turbidity-100m",
        "cdom-100m",
        "chlorophyll-a-100m-mask",
        "turbidity-100m-mask",
        "cdom-100m-mask",
    ]

    def plot(self, sample):
        # Reorder and rescale the image
        image = sample["image"].permute(1, 2, 0)

        # Plot the image
        plt.imshow(image)


class RioNegroMeteorological(RasterDataset):
    filename_glob = "*_meteorological.tif"
    filename_regex = r"(?P<date>\d{8})"
    date_format = "%Y%m%d"
    is_image = True
    separate_files = False
    all_bands = [
        "air_temperature_min",
        "air_temperature_max",
        "air_temperature_mean",
        "cloud_coverage_mean",
        "precipitation_sum",
        "radiation_min",
        "radiation_max",
        "radiation_mean",
        "relative_humidity_min",
        "relative_humidity_max",
        "relative_humidity_mean",
        "u_wind_mean",
        "v_wind_mean",
    ]

    def plot(self, sample):
        # Reorder and rescale the image
        image = sample["image"].permute(1, 2, 0)
        image = image[:, :, 0]

        # Plot the image
        plt.imshow(image)


class RioNegroWaterTemperature(RasterDataset):
    filename_glob = "*_water_temperature.tif"
    filename_regex = r"(?P<date>\d{8})"
    date_format = "%Y%m%d"
    is_image = True
    separate_files = False
    all_bands = ["water_temperature", "water_temperature_mask"]

    def plot(self, sample):
        # Reorder and rescale the image
        image = sample["image"].permute(1, 2, 0)
        image = image[:, :, 0]

        # Plot the image
        plt.imshow(image)


class RioNegroData(torch.utils.data.Dataset):  # type: ignore[attr-defined]
    """
    Rio Negra dataset definition for PyTorch.

    The original biological dataset is used as the ground truth. It is used to retrieve
    the corresponding samples from the other datasets.

    The input data contains the following features, in this order:
    - Chlorophyll-a (processed)
    - Turbidity (processed)
    - CDOM (processed)
    - Water temperature (processed)
    - Air temperature min
    - Air temperature max
    - Air temperature mean
    - Cloud coverage mean
    - Precipitation sum
    - Radiation min
    - Radiation max
    - Radiation mean
    - Relative humidity min
    - Relative humidity max
    - Relative humidity mean
    - U wind mean
    - V wind mean

    Masks are provided to indicate which pixels are unmodified in the processed datasets
    for the following features: Chlorophyll-a, Turbidity, CDOM, Water temperature.

    Args:
        root (str): The root directory of the dataset.
        reservoir (str): The reservoir to use (e.g. 'palmar').
        window_size (int): The number of samples to use as input.
        prediction_horizon (int): The number of days to predict ahead.
        input_size (int): The size of the image.

    Returns:
        A tuple of (images, masks, targets) where: image is a batch of samples of shape
        (batch_size, window_size, num_bands, input_size, input_size), mask is a batch of
        shape (batch_size, window_size, 4, input_size, input_size) and target is a batch
        of samples of shape (batch_size, num_bands, input_size, input_size).

        The input samples are the <window_size> number of sequential samples
        <prediction horizon> timesteps before the target sample. E.g. for a prediction
        horizon of 4 days and a window size of 3, the input samples are samples (t-6),
        (t-5) and (t-4), where the target sample is (t).
    """

    def __init__(
        self,
        root: str,
        reservoir: str,
        window_size: int,
        prediction_horizon: int,
        input_size: int = 224,
    ):
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.input_size = float(input_size)

        # Load separate datasets
        print("Loading datasets...")

        s = time.time()
        data_path = os.path.join(root, "biological", reservoir)
        self.data_bio_unprocessed = RioNegroBiological(root=data_path)
        e = time.time() - s
        print("Loaded unprocessed biological dataset in {:.2f} seconds".format(e))

        s = time.time()
        data_path = os.path.join(root, "biological_processed", reservoir)
        data_bio = RioNegroBiological(root=data_path)
        e = time.time() - s
        print("Loaded processed biological dataset in {:.2f} seconds".format(e))

        s = time.time()
        data_path = os.path.join(
            root, "physical_processed/water_temperature", reservoir
        )
        data_water_temp = RioNegroWaterTemperature(root=data_path)
        e = time.time() - s
        print("Loaded processed water temperature dataset in {:.2f} seconds".format(e))

        s = time.time()
        data_path = os.path.join(root, "meteorological_processed")
        data_meteo = RioNegroMeteorological(root=data_path)
        e = time.time() - s
        print("Loaded processed meteorological dataset in {:.2f} seconds".format(e))

        # Make intersection of datasets. TorchGeo will automatically convert resolutions
        # to be compatible.
        dataset_biophys = data_bio & data_water_temp
        self.dataset = dataset_biophys & data_meteo

        # Descriptions of bands.
        self.all_bands = (
            data_bio.all_bands[:3]
            + [data_water_temp.all_bands[0]]
            + data_meteo.all_bands[2:4]
            + [data_meteo.all_bands[7]]
            + data_meteo.all_bands[10:]
        )

        # We sample the ground-truth from the unprocessed bio data which has unmodified
        # values. We need to make sure that we sample within the range of all datasets.
        tdelta = timedelta(days=window_size + prediction_horizon)
        tmin = [
            self.data_bio_unprocessed.bounds.mint,
            (datetime.fromtimestamp(data_bio.bounds.mint) + tdelta).timestamp(),
            (datetime.fromtimestamp(data_meteo.bounds.mint) + tdelta).timestamp(),
            (datetime.fromtimestamp(data_water_temp.bounds.mint) + tdelta).timestamp(),
        ]
        tmax = [
            self.data_bio_unprocessed.bounds.maxt,
            data_bio.bounds.maxt,
            data_meteo.bounds.maxt,
            data_water_temp.bounds.maxt,
        ]

        self.roi = BoundingBox(
            self.data_bio_unprocessed.bounds.minx,
            self.data_bio_unprocessed.bounds.maxx,
            self.data_bio_unprocessed.bounds.miny,
            self.data_bio_unprocessed.bounds.maxy,
            np.max(np.array(tmin)),
            np.min(np.array(tmax)),
        )

    def __getitem__(self, bbox):
        # Get bio sample as ground truth.
        gt = self.data_bio_unprocessed.__getitem__(bbox)["image"][0, :, :]

        # We then retrieve the corresponding samples from the processed datasets
        # according to the bounding box of the bio sample and the defined window size
        # and prediction horizon.

        # Convert unix timestamp float to datetime object.
        mint = datetime.fromtimestamp(bbox.mint)
        maxt = datetime.fromtimestamp(bbox.maxt)

        # Set time to 00:00:00 and 23:59:59.
        mint = mint.replace(hour=0, minute=0, second=0, microsecond=0)
        maxt = maxt.replace(hour=23, minute=59, second=59, microsecond=999999)

        sample_list = []
        # Iterate from current - (window_size + prediction_horizon) to current - prediction_horizon.
        for i in reversed(
            range(self.prediction_horizon, self.prediction_horizon + self.window_size)
        ):
            # Create bbox for the current time step.
            bbox = BoundingBox(
                mint=(mint + timedelta(days=-i)).timestamp(),
                maxt=(maxt + timedelta(days=-i)).timestamp(),
                minx=bbox.minx,
                maxx=bbox.maxx,
                miny=bbox.miny,
                maxy=bbox.maxy,
            )

            # Get samples.
            sample_list.append(self.dataset.__getitem__(bbox)["image"])

        x = torch.stack(sample_list)

        # Split images and masks into separate tensors.
        # [x[:, :3, :, :], x[:, 6, :, :].unsqueeze(1)], dim=1
        # Using bands [0, 1, 2, 3, 6, 7, 8, 11, 14, 15, 16].
        x_image = torch.cat(
            [
                x[:, :3, :, :],
                x[:, 6, :, :].unsqueeze(1),
                x[:, 10:13, :, :],
                x[:, 15, :, :].unsqueeze(1),
                x[:, 18:, :, :],
            ],
            dim=1,
        )
        x_mask = torch.cat([x[:, 3:6, :, :], x[:, 7, :, :].unsqueeze(1)], dim=1).bool()

        # Assert x_mask only contains 0 and 1.
        assert torch.all(torch.logical_or(x_mask == 0, x_mask == 1))

        return x_image, x_mask, gt

    def __len__(self):
        return len(self.dataset)

    def __add__(self, _):
        raise NotImplementedError

    def plot(self, batch):
        sample = batch[0][0, 0, :, :, :].numpy()
        titles = self.all_bands

        plt.figure(figsize=(20, 5))
        for i in range(sample.shape[0]):
            plt.subplot(2, 9, i + 1)
            plt.title(titles[i])
            plt.imshow(sample[i, :, :], cmap="viridis")
            plt.axis("off")
        plt.show()
