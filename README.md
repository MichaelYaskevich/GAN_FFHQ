# GAN FFHQ

Generative adversarial network (GAN) trained on Flickr-Faces-HQ (FFHQ) for face generation. \
After training we count leave-one-out-1-NN classifier accuracy of the GAN. \
We also use TSNE for generated and real images with visualization.

## GAN architecture
You can find architecture in scr/gan_architecture/GanModel.py.\
GAN can be constructed for different input image sizes 64x64, 128x128, etc.

## Launch examples
There are 2 model: train and eval
1) Help for train and eval\
...\GAN_FFHQ>python main.py -h
2) Help for train\
...\GAN_FFHQ>python main.py train -h
3) Help for eval\
...\GAN_FFHQ>python main.py eval -h

You can find prepared weights in scr/resources/GanModel128x128.zip

For training you have to load dataset as images to \
scr/resources/faces_dataset_small/faces_dataset_small

Train\
...\GAN_FFHQ>train 64 2 0.0001 path_to_save_training_results

Eval\
...\GAN_FFHQ>eval 128 50 path_to_save_eval_results scr/resources/GanModel128x128.zip