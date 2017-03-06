ISZ = 128
smooth = 1e-12
dice_coef_smooth = 1
batch_size = 4
num_epoch = 10
validation_patches = 600
train_patches = 6000
image_size = 1024  # MUST BE DIVISIBLE BY ISZ
image_depth = 20

# optimizer parameters
learning_rate = 0.0001
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-07

# misc
generate_label_masks = False
# test_nums = [8, 10, 12, 6, 17]
test_nums = []
