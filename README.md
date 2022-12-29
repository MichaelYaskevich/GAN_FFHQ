# GAN FFHQ

Generative adversarial network (GAN) trained on Flickr-Faces-HQ (FFHQ) for face generation. \
After training we count leave-one-out-1-NN classifier accuracy of the GAN. \
We also use TSNE for generated and real images with visualization.

## GAN architecture:
You can find architecture in scr/gan_architecture/GanModel.py.\
GAN can be constructed for different input image sizes 64x64, 128x128, etc.
