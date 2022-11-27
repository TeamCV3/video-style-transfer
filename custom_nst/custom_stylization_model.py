import tensorflow as tf
import numpy as np
import PIL.Image

class StyleContentModel(tf.keras.models.Model):
    """Model that uses VGG19 to compute style and content tensors."""

    def __init__(self, style_layers, content_layers):
        """ Initializes the model and parameters.
            Args:
                style_layers: List of layer names to use for style.
                content_layers: List of layer names to use for content. 
        """

        super(StyleContentModel, self).__init__()
        self.vgg = self.vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def vgg_layers(self, layer_names):
        """ Creates a vgg model that returns a list of intermediate output values.
            Args:
                layer_names: List of layer names to use for style.
            Returns:
                Model that takes image inputs and outputs the style and content intermediate layers.
        """

        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False

        # make a model that returns the intermediate layer outputs
        outputs = [vgg.get_layer(name).output for name in layer_names]
        model = tf.keras.Model([vgg.input], outputs)

        return model

    def gram_matrix(self, input_tensor):
        """ Computes the gram matrix of an input tensor.
            Args:
                input_tensor: Tensor to compute the gram matrix of.
            Returns:
                Gram matrix of the input tensor.
        """

        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
        return result / (num_locations)

    def call(self, inputs):
        """ Computes the style and content tensors for the given input.
            Args:
                inputs: Input tensor to compute the style and content tensors for. Expects float input in [0,1]           
            Returns:
                Style and content tensors.
        """

        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                        outputs[self.num_style_layers:])

        style_outputs = [self.gram_matrix(style_output)
                        for style_output in style_outputs]

        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value
                    for style_name, value
                    in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}


class CustomStylizationModel:
    """Custom style transfer model"""

    def __init__(
        self,
        content_image_path,
        style_image_path,
        content_layers=['block5_conv2'],
        style_layers=['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1'],
        style_weight=1e-2,
        content_weight=1e4,
        total_variation_weight=30,
        n_iter=1000,
        opt=tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
    ):
        """ Initializes the model and parameters.
            Args:
                content_image_path: Path to the content image.
        """

        self.content_layers = content_layers
        self.style_layers = style_layers
        self.extractor = StyleContentModel(self.style_layers, self.content_layers)
        self.opt = opt
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.total_variation_weight = total_variation_weight
        self.n_iter = n_iter

        self.content_image_path = content_image_path
        self.style_image_path = style_image_path
    
    def compose(self):
        """ Composes stylized image from the content and style images.
            Returns:
                Stylized image.
        """

        content_image = self.load_img(self.content_image_path)
        style_image = self.load_img(self.style_image_path)

        self.style_targets = self.extractor(style_image)['style']
        self.content_targets = self.extractor(content_image)['content']

        image = tf.Variable(content_image)

        for _ in range(self.n_iter):
            self.train_step(image)

        return self.tensor_to_image(image)


    def style_content_loss(self, outputs):
        """ Computes the total loss.
            Args:
                outputs: Style and content tensors.
            Returns:
                Total loss.
        """
        style_outputs = outputs['style']
        content_outputs = outputs['content']
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-self.style_targets[name])**2) 
                            for name in style_outputs.keys()])
        style_loss *= self.style_weight / len(self.style_layers)

        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-self.content_targets[name])**2) 
                                for name in content_outputs.keys()])
        content_loss *= self.content_weight / len(self.content_layers)
        loss = style_loss + content_loss
        return loss

    def load_img(self, path_to_img):
        """ Loads an image from the given path.
            Args:
                path_to_img: Path to the image.
            Returns:
                Image tensorflow tensor.
        """

        max_dim = 512
        img = tf.io.read_file(path_to_img)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)

        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim

        new_shape = tf.cast(shape * scale, tf.int32)

        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]
        return img

    @tf.function()
    def train_step(self, image):
        """ Performs a single optimization step on the image.
            Args:
                image: Image to perform the training step on.
        """

        with tf.GradientTape() as tape:
            outputs = self.extractor(image)
            loss = self.style_content_loss(outputs)
            loss += self.total_variation_weight * tf.image.total_variation(image)

        grad = tape.gradient(loss, image)
        self.opt.apply_gradients([(grad, image)])
        image.assign(
            tf.clip_by_value(
                image, clip_value_min=0.0, clip_value_max=1.0
            )
        )
    
    def tensor_to_image(self, tensor):
        """ Converts a tensor to an image.
            Args:
                tensor: Tensor to convert to an image.
            Returns:
                PIL Image.
        """

        tensor = tensor*255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor)>3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return PIL.Image.fromarray(tensor)
