import os

cwd = os.getcwd()
cwd = cwd + '/..'

aesthetic_model = cwd+'/models/aesthetic_model_40_500_v27.hdf5'
technical_model = cwd+'/models/technical_model_30_500_v6.hdf5'

angle_model       = cwd+'/models/mobilenet_Angle_50_52_4_64.hdf5'
white_model       = cwd+'/models/mobilenet_White_50_70_4_64.hdf5'
contrast_model    = cwd+'/models/mobilenet_Contrast_50_50_15_64.hdf5'
exposure_model    = cwd+'/models/custom_model_Exposure_50_24_4_64.hdf5'
straight_model    = cwd+'/models/mobilenet_Straight_50_60_4_64.hdf5'
sharpness_model   = cwd+'/models/mobilenet_Sharpness_50_70_4_64.hdf5'
brightness_model  = cwd+'/models/mobilenet_Brightness_50_50_5_64.hdf5'
distortion_model  = cwd+'/models/mobilenet_Distortion_50_40_4_64.hdf5'
saturation_model  = cwd+'/models/mobilenet_Saturation_50_70_4_64.hdf5'
composition_model = cwd+'/models/mobilenet_Composition_50_50_4_64.hdf5'
