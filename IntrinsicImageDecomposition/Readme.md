
#Intrinsic Imaging
*Prateek Gulati*


**Abstract**
In this project, I replicate Das et al.’s work for Intrinsic Image Decomposition which uses an encoder-decoder architecture. In the training process, a shared encoder is employed with separate decoders which have a layer of attention between them. I also replicate Liu et al.’s work on unsupervised intrinsic image decomposition which directly learns the latent feature of reflectance and shading from unsupervised and uncorrelated data. Later I compare the results from both models based on visual results. To further compare the model performances, I use image processing techniques and compare their effect on applying a) directly to the image vs b) applying to an intrinsic property of the image and combining later. These techniques include hue, gamma correction, and removing objects by manually editing the image.

## 1. Introduction
In this project, I address the problem of Intrinsic Image Decomposition, which is an essential task in computer vision. Given an input image, the goal is to infer the reflectance and shading of the scene. For this, I refer to two papers primarily: Liu et al.’s work on Intrinsic Image Decomposition from a Single Image using Unsupervised Learning (USI3D); and Das et al.’s work on Photometric Invariant Edge Guided Network for Intrinsic Image Decomposition (PIENet). 

The first work, PIENet focuses on gradients based on illumination invariant descriptors or CCRs. They only depend on albedo changes. They propose a hierarchical CNN that includes global and local refinement layers. The global layer eliminates negative illumination transitions and reduces shading reflectance misclassification and the local layer eliminates the hard negatives. Spatial attention layers are included to focus on image areas containing hard negatives. The code for this is borrowed from Partha Das’s GitHub. 

The next work, USI3D estimates reflectance and shading from a natural image as a transferring image style. They collect three unlabeled and uncorrelated samples for each set and use an unsupervised learning method to learn the style of a natural image, reflectance, and shading from it. Then they apply auto-encoder and GAN to transfer the image to the desired style while preserving the underlying content. The code for this is referred from Yunfei Liu’s GitHub repository.

For the evaluation, the results of these methods are compared with each other. To further analyze the reflectance and shading of a natural image, a transformation technique is applied to the original image vs a corresponding intrinsic property and their results are compared.
## 2. Dataset
The USI3D model uses four datasets, ShapeNet intrinsic dataset, MPI-Sintel benchmark, MIT intrinsic dataset, and Intrinsic Images in the Wild benchmark. PIENet also uses four datasets, out of which the MIT, Sintel, and IIW datasets are the same. It also uses the NED dataset.  
## 3. Methods
### 3.1 PIENet
`	`The PIENet’s architecture has a) a shared encoder for CCR and image which learns illumination invariant reflectance, and illumination and reflectance features respectively; b) a Linked Edge Decoder to learn a relational representation of the cues; c) an Unrefined Decoder which takes input from a Linked edge decoder passed to a layer of attention and hence able to focus on global consistencies; d) Local Refinement Module which takes concatenated input from different decoders, calibrates it, encodes it to embedding space and separate decoders to generate refined reflectance and shading respectively.
### 3.2 USI3D
The USI3D architecture can be stated as a) Content-sharing which has an encoder to extract content code which is passed to generators to generate a decomposition layer for reflectance and shading respectively, b) a mapping module to infer prior code from the image set, c) autoencoders which has bidirectional reconstruction constraints which enable reconstruction in both directions. 
## 4 Experiments
For the experimentation, I have used an 8 CPU with a v100 Intel GPU enabled with Cuda/11.2 on a Linux operating system. 
### 4.1 PIENet
As a starting point, I referred to Partha Das’s GitHub repository which has the implementation on the PIENet paper written in PyTorch. After the required modification, the code works well and yields reflectance and shading of a given image. To replicate the results, use the model weights from real\_world\_model.t7. This model doesn’t generate good results with different image sizes. So, the input to the model is an image of size 256x256, which is the size it’s trained on. For evaluation, the images need to be resized to match the dimension.  

![figure1](/IntrinsicImageDecomposition/assets/figure1.png)

**Figure 1**: *Reflectance and Shade from PIENet. \*PS: The labeling has been swapped in this snippet*
### 4.2 USI3D
For this, I refer to Yunfei Liu’s GitHub repository which has a PyTorch implementation of the USI3D paper. The best result of this model is using the IIW dataset. The model trained with just MPI or ShapeNet doesn’t have desired results visually. For the IIW dataset, a base learning rate of 0.5 is used with Adam optimizer and a weight decay of 10-4. The last layer of the generator and discriminator has 64 features for an image as the batch size is 1. For better visual results, the individual pixel value from shading is halved. This prevents fading of the image when regenerated. 

![figure2](/IntrinsicImageDecomposition/assets/figure2.png)

**Figure 2**: *Reflectance and Shade from PIENet. \*PS: The labelling has been swapped in this snippet*
### 4.3 **Visualization**
For the results, we first compare some real-world samples which are unseen for both models. We see that both generate slightly different decompositions, USI3D seems to have better reflectance maps. 

Next, we transform the image, first, applying on the original image, and second, by applying on any one of the decompositions and combing later. The first transformation used is hue. A random hue value is selected and applied on the original image and compared with hue transformation on the reflectance and combined with the original shading. Both models give very similar results, which means a majority of the color is correctly captured by the reflectance. For the second transformation, we use gamma correction, only this time we apply it to shading and combining with reflectance to compare with the original image. For the same image, the results aren't identical but more vibrant than the gamma correction on the original image respectively.  

![figure3](/IntrinsicImageDecomposition/assets/figure3.png)

**Figure 3**: *Random Hue on PIENet. Left: Original Image, Middle: Hue on original image, Right: (Hue on Reflectance) x Shading*

![figure4](/IntrinsicImageDecomposition/assets/figure4.png)

**Figure 4**: *Random Hue on USI3D. Left: Original Image, Middle: Hue on original image, Right: (Hue on Reflectance) x Shading*

![figure5](/IntrinsicImageDecomposition/assets/figure5.png)

**Figure 5**: *Gamma Correction on PIENet. Left: Original Image, Middle: Gamma Correction on original image, Right: (Gamma Correction on Shading) x Reflectance*



![figure6](/IntrinsicImageDecomposition/assets/figure6.png)

**Figure 6**: *Gamma Correction on USI3D. Left: Original Image, Middle: Gamma Correction on original image, Right: (Gamma Correction on Shading) x Reflectance*

Lastly, on one of the test images, I photoshop some design out of the reflectance map and combine with shading. Since the altered part is all on the same flat surface, ideally adding shading to it should not change anything. But it’s a hard problem, and both the model have some residue of the structure in the shadow. 

![figure7](/IntrinsicImageDecomposition/assets/figure7.png)

**Figure 7**: *Photoshop on PIENet. Left to Right: Original Image, Reflectance, Photoshopped Reflectance, Photoshopped Reflectance with original shading*

![figure8](/IntrinsicImageDecomposition/assets/figure8.png)

**Figure 8**: *Photoshop on USI3D. Left to Right: Original Image, Reflectance, Photoshopped Reflectance, Photoshopped Reflectance with original shading*
## 5 Conclusion
In this work, I have experimented with two intrinsic image decomposition methods that use very advanced state-of-the-art architectures. Both the models perform well in most scenarios but have their downsides too. When testing with real-world complex images, both models fall short of the ideal result. Despite advanced architectures in this field, there is still scope for improvement to the image decomposition problem when dealing with more complex settings.      

## References
Partha Das, Sezer Karaoglu, & Theo Gevers (2022). PIE-Net: Photometric Invariant Edge Guided Network for Intrinsic Image Decomposition. In IEEE Conference on Computer Vision and Pattern Recognition, (CVPR).

Liu, Y., Li, Y., You, S., & Lu, F. (2020). Unsupervised Learning for Intrinsic Image Decomposition from a Single Image. In CVPR.

S. Baslamisli, T. T. Groenestege, P. Das, H. A. Le, S. Karaoglu, and T. Gevers. Joint learning of intrinsic images and semantic segmentation. In ECCV, 2018

D. J. Butler, J. Wulff, G. B. Stanley, and M. J. Black. A naturalistic open source movie for optical flow evaluation. In ECCV, 2012

R. Grosse, M. K. Johnson, E. H. Adelson, and W. T. Freeman. Ground truth dataset and baseline evaluations for intrinsic image algorithms. In ICCV, 2009

S. Bell, K. Bala, and N. Snavely. Intrinsic images in the wild. ACM TOG, 2014

Partha Das <https://github.com/Morpheus3000/>

Yunfei Liu <https://github.com/DreamtaleCore/>

`	`7
