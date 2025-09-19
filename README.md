# unet_subsurface_velocities
This project uses the U-Net architecture—originally developed for image segmentation—for predicting subsurface ocean dynamics from surface variables, specifically sea surface height (SSH) and sea surface temperature (SST)

## 🧪 Testing the Training Script with Dummy Data

A dummy dataset is included in the `dummy_dataset/` folder to allow quick testing of the training pipeline without requiring real NetCDF input files.

To run the training script using the dummy data, simply execute:

```bash
python3 train_unet.py --dummy
