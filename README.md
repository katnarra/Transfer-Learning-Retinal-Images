The aim of the project was to improve the multi-label retinal image classification pipeline to better detect diabetic retinopathy, glaucoma, and age-related macular degeneration from fundus images.
Model performance was optimized by fine-tuning pre-trained CNN backbones. Classification accuracy was improved through evaluation of appropriate loss functions and ensemble learning. Below is a short report of the project.


# INTRODUCTION
Medical datasets are often small as obtaining images and annotating them is time consuming and difficult. To address
this limitation, transfer learning is a technique where a pretrained model provides knowledge that can be leveraged when
fine-tuning a model with a smaller dataset. Two different pretrained backbones, EfficientNet and ResNet-18 are fine-tuned
for multi-label retinal image classification, focusing on three
major conditions: Diabetic Retinopathy (DR), Glaucoma (G),
and Age-related Macular Degeneration (AMD). The performance metrics used to evaluate the model performances include, but are not limited to, precision, recall, and F1-score.
The project aims to improve multi-label retinal image classification by fine-tuning models with different techniques.


# Methods
The ODIR dataset with 800 images were used for training, whereas the validation set consisted of 200 images. 
The models were evaluated on a locally run offsite test set of 300 images as well as a separate onsite test set with 250 images. 
THe evaluation of performance was done based on accuracy, precision, recall, F1-score, and kappa.

## Transfer Learning
Transfer learning was performed to two pretrained models, EfficientNet and ResNet18,
that have been trained with large-scale datasets ADAM,
REFUGE2, and APTOS.

In this project, transfer learning was conducted with three
different setups to see how fine-tuning affected the performance of the resulting models. First, the pretrained backbones from 
EfficientNet and ResNet18 were used directly on the ODIR test set without any fine-tuning to establish a baseline. The second 
transfer learning was done with a frozen backbone, only allowing the fine-tuning of the classifier with the ODIR training set. 
Here, a learning rate of 1e−3 was used along with a batch size of 32 and number of epoch being 20 for both backbones. 
Finally, full fine-tuning updated both the pretrained backbone and the classifier on ODIR training set.

## Loss Function
The class frequencies of diabetic retinopathy, glaucoma, and age-related macular degeneration are quite unbalanced
in the ODIR dataset with DR dominating the distribution (63%). Glaucoma and AMD cover about 20% and 17% of
the dataset, respectively. Due to this strong class imbalance, the model is likely to overpredict DR while underestimating
cases of glaucoma and AMD. To see how the balancing of class frequencies affects the performance, two different loss
functions were implemented to replace the originally used Binary Cross-Entropy, focall loss and class-balanced loss.

In this project, the class importance weights and the focusing parameter were set to α = [0.4, 0.7, 0.75], γ = 2, respectively. 
The values for α come from the class frequencies: the higher frequency, the lower importance.
The chosen weights for CB loss were 0.59, 4, and 4.88 for DR, G, and AMD, respectively.

## Ensemble Learning
There are many approaches on how the models can be combined, but for this project, weighted average was
chosen because of its simple and fast implementation. For example, there is no need to train any new models as ensemble
learning uses existing models. Each model produced probabilities of the image belonging to the three disease classes.
These probabilities were averaged across two models using the weights, and these weighted-average probabilities were
used to make the final predictions. Initially, the weights were set to be equal, but the eventual choice was to use a weight
of 0.4 for the lower-performing model and 0.6 for the better-performing model.

# Results
The results are covered briefly. 

The results show that ResNet-18 is the better and more robust choice when the whole backbone is updated, but when timely efficiency
is needed, fine-tuning only the EfficientNet classifier is the better option. This is because fully fine-tuning the EfficientNet backbone
does not bring any significant advantages compared to only fine-tuning the classifier, whereas fully fine-tuning the ResNet18 backbone
further improves the results compared to only fine-tuning the classifier.

Both loss functions improved the performance quite well. The prediction of glaucoma seems to remain the most accurate across
both loss functions. Originally, glaucoma’s recall dropped when the pretrained EfficientNet was fine-tuned, but
by using other loss functions it remained more stable.

By combining two models, the predictive performance was further improved. These results show that by using
the weights 0.4 and 0.6, the performance across the three disease types was stabilized, resulting in a better overall performance on the onsite test set. 
Notably, by combining ResNet-18 with EfficientNet, the results were even better. With the highest obtained offsite F1-score (81.4%) these different network 
architectures seem to endorse each other’s capabilities while reducing individual model errors.

# Discussion
Overall, these results highlight the simpler architecture of ResNet-18 compared to EfficientNet. With fewer parameters,
it seems to require more training to learn the specific spatial features of this dataset, whereas EfficientNet was more
prone to overfitting. Using focal loss or class-balanced loss to account for class imbalances proved to be a good method
to increase model performance. In addition, using weighted average pushed the performance even further, allowing quite
good predictive power. To summarize, while EfficientNet offers high efficiency with minimal tuning, and ResNet-18
superior robustness when fully trained, specialized loss functions and ensembling proved more effective than attention
mechanisms for improving generalization on this limited dataset.

## Note!
Attention mechanisms were also tried out with the goal of improving performance. However, during the time span I had for this project,
I was not able to get them to work properly/improve the results.

