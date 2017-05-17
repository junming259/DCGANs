# DCGANs
Implement DCGANs from paper https://arxiv.org/pdf/1511.06434.pdf

During the implementation of DCGANs, some points needs to be taken cared:

1. Instead of using tf.contrib.layers.batch_norm, I implement my own batch
normalization layer. Take care the difference between training mode and
testing mode.

2. Sharing variables is very important in this network, because I need to
define two discriminator. One is to discriminate real data, the other is to
discriminate generated image.

3. The original data type of cifar10 is np.uint8, during loading data, I
convert it into np.float32 within range -1 to 1.

4. When displaying or saving image, we need to de-proccess image back to range
from 0 to 255 with dtype np.uint8.

5. The result of pre batch normalization is better than the result of post
batch normalization. You can compare results in pre-batch or post-batch
folder.
