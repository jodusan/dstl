ISZ = 128
smooth = 1e-12
dice_coef_smooth = 1
batch_size = 16
num_epoch = 10
validation_patches = 1000
train_patches = 6000 
image_size = 1024  # MUST BE DIVISIBLE BY ISZ
image_depth = 20

image_scale_max = 1.2
image_scale_min = 0.8

# optimizer parameters
learning_rate = 0.0002
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-07

# misc
generate_label_masks = False
# test_nums = [8, 10, 12, 6, 17]
test_nums = []

assert image_size % ISZ == 0, "Image size must be divisible by ISZ"
assert image_scale_max <= (image_size / ISZ), "Max image scale must be less than or equal to image_size divided by ISZ"
