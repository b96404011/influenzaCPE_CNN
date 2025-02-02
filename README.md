# Differentiation of Cytopathic Effects (CPE) Induced by Influenza Virus Infection Using Deep Convolutional Neural Networks (CNN)

## abstract

Cell culture remains as the golden standard for primary isolation of viruses in clinical specimens. In the current practice, researchers have to recognize the cytopathic effects (CPE) induced by virus infection and subsequently use virus-specific monoclonal antibody to confirm the presence of virus. Considering the broad applications of neural network in various fields, we aimed to utilize convolutional neural networks (CNN) to shorten the timing required for CPE identification and to improve the assay sensitivity. Based on the characteristics of influenza-induced CPE, a CNN model with larger sizes of filters and max-pooling kernels was constructed in the absence of transfer learning. A total of 601 images from mock-infected and influenza-infected MDCK cells were used to train the model. The performance of the model was tested by using extra 400 images and the percentage of correct recognition was 99.75%. To further examine the limit of our model in evaluating the changes of CPE overtime, additional 1190 images from a new experiment were used and the recognition rates at 16 hour (hr), 28 hr, and 40 hr post virus infection were 71.80%, 98.25%, and 87.46%, respectively. The specificity of our model, examined by images of MDCK cells infected by six other non-influenza viruses, was 100%. Hence, a simple CNN model was established to enhance the identification of influenza virus in clinical practice.


The data in the github was divided. Original input data was merged form. You can also download non-divided data from below.
[dataset download](https://ppt.cc/f8uK5x)
