# Week 7 - Style transfer and Generative Adversarial Networks
This week we were introduced to style transfer and generative adversarial networks. Whilst I have covered both in detail before, this was sueful learning exercise for me to understand tensorflow - as most of my ML experience has been using the pytorch library.

## Style transfer
Using a pretrained model via `tensforflow_hub` I was able to transfer the style from one image to the content of another. This can be seen below.
![alt](img/style2.png)
\
\
I additionally opted to carry out some experiments on the preprocessing of the style image. The default involved (after resizing to 256x256) applying average pooling with kernel size 3 and stride 1 (both directions). Average pooling is a technique used to downsample an image into a smoother format. The kernel size affects how averaged the image will be and the stride affects how many pixels the kernel moves along in the image; a kernel and stride size of 1 will return the same image.
#### Default (k=3, s = 1)
![](img/style1.png)
#### k=1, s = 1
![](img/style1_k1.png)
#### k=5, s = 1
![](img/style1_k2.png)
#### k=3, s = 2
![](img/style1_s2.png)
#### k=3, s = 3
![](img/style1_s3.png)
