ISZ = 160
smooth = 1e-12
dice_coef_smooth = 1
batch_size = 4
num_epoch = 4
validation_patches = 400
train_patches = 1800
image_size = 800  # MUST BE DIVISIBLE BY ISZ
image_depth = 8

# optimizer parameters
learning_rate = 0.0001
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-07

# misc
generate_label_masks = False
#test_nums = [8, 10, 12, 6, 17]
test_nums = []
