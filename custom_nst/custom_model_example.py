from custom_stylization_model import CustomStylizationModel
import tensorflow as tf

# create a custom NST model and initialize parameters
model = CustomStylizationModel(
    'content.jpg',
    'style.jpg',
    content_layers=['block5_conv2'],
    style_layers=['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1'],
    style_weight=1e-2,
    content_weight=1e4,
    total_variation_weight=30,
    n_iter=1000,
    opt=tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
)

# compose resulting image
result = model.compose()

# save resulting image
result.save('result.jpg')