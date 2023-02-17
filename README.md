# Apple_Plant_Disease_Predictor

![image](https://user-images.githubusercontent.com/96066261/219586092-15f8096b-2116-45a9-ab78-7bb947991052.png)       ![image](https://user-images.githubusercontent.com/96066261/219586167-683db876-e94a-4aee-8921-1d783eff9452.png)     ![image](https://user-images.githubusercontent.com/96066261/219586225-8c0439b2-6a16-45be-a58b-bf4bb1abb120.png)

## Problem Statement
India is an agricultural country with bounty of all types of crops as well as fruits grown across the nation. One of the fruits which is mostly everyone' s favorite i.e., Apple plant.
The apples of Kashmir are famous worldwide for their delicacy and juiciness. Thus, the main aim is to build a model consisting of such capability as to precicely discover or
predict the diseases in the apple plant. Some of the common diseases are Apple rot, Scab leaves and Leaf blotch.


![download](https://user-images.githubusercontent.com/96066261/219589808-ff98f624-87e5-4f03-a62b-4c3ab46c5325.jpeg) ![download (1)](https://user-images.githubusercontent.com/96066261/219589920-d951fcef-b3df-447a-811d-6a5faf031e9d.jpeg)![download (2)](https://user-images.githubusercontent.com/96066261/219589962-d08f314a-5d7d-4276-8299-108523aae5d7.jpeg)

The above are some examples of such diseases.
The dataset consists of leaf images of various apple plants from Kashmir and fortunaltely this is the latest dataset available from Kashmir. The dataset used comprises different diseases which are found in apple plants and as well as there are healthy leaf images.

![image](https://user-images.githubusercontent.com/96066261/219602900-04908c49-da2f-41f9-82ef-a968379e2693.png)


You can easily download the dataset from the link:- https://drive.google.com/drive/folders/1FWsZxnEGdcUfMjhAXwCP4KKkpViABrGI?usp=share_link

## Why is there a need for such a model?
### Existing System
The main existing system for the plants or specifically the apple plants is the manual method, where labours and farm workers check each and every apple plant. 
This boosts up the costs for the maintenance and it is also highly unreliable due to manual works which results in low precision. 
There are also possibilities that the workers may neglect their duties or some plants which further decreases the quality production. 
The main concern for any farmer is the quality and quantity of their crops or any other fruits they are cultivating. Thus, the traditional method which has been used for centuries is becoming very outdated due to population burst throughout the nation. 

#### Disadvantages
<ol type = "i">
<li>	Time Consuming and less effective: Manual methods are very time consuming. </li>
<li>	Unreliability: Traditional methods are unreliable. </li>
<li>	Very high costs and less production: The costs for any manual labour are very high and the production decreases if there are mistakes during the checking of the plants. </li>
<li>  Lesser Quality: The diseased plants produce low quality crops or fruits and these products rot very quickly and are sold for cheap prices. </li>
<li>	More Waste of Useful Land: The lesser production may not be seen in small farms but when these irregularities appear on large scale, there are drastic results such as shortage in the food supplies for certain products. Thus, increasing the prices. </li>
</ol>

![download (3)](https://user-images.githubusercontent.com/96066261/219596852-4ae0529a-a203-4d1a-a7ea-43e44f941b22.jpeg)     ![download (4)](https://user-images.githubusercontent.com/96066261/219596890-78e87aae-d66e-46a5-aefc-61650cf2dfec.jpeg)

### Proposed System

The proposed system contains the use of Machine Learning and Deep Learning techniques. The catch is that when a plant becomes sick, the leaves are the first to exhibit indications, therefore the model can anticipate by analysing photos of the leaves. A paradigm or framework like this will assist us in better diagnosing illnesses. 
The Plant Disease Predictor will be made up of two primary components:
1. A feature extractor 
2. A Classifier
The feature extractor takes visual information from input photos and delivers them to the training model. During training, the model use prediction algorithms to analyse the supplied photos and forecast whether they are contaminated or not. Based on the data images collected, our plant predictor will detect the infected with better accuracy and precision.
The findings throughout this project will help farmers to quickly take countermeasures against the contaminated and infected plants and trees.

![216238056-7d1ce807-c5b2-4dec-a554-60846b978474](https://user-images.githubusercontent.com/96066261/219598586-b2f61da9-0e99-4cd3-8130-641b5aeed36c.png)


#### Advantages
<ol type = "i">
<li> Cost Reduction: The manual methods or traditional methods required labours, but our proposed will only require one time investment and very less maintenance cost. </li>
<li> Less Time Consuming: With the help of only images, we can predict if a plant is infected or not. </li>
<li> Reliability: The model has a high reliability with all the machine learning and deep learning techniques. </li>
<li> Quality: With the plants being saved at an earlier stage the quality of the crop increases which also results in the increase in production. </li>
</ol>

## SOFTWARE REQUIREMENT
**Python**

Python is an interpreted, high-level and general-purpose programming language. Created by Guido van Rossum and first released in 1991, Python’s design philosophy emphasizes code readability with its notable use of significant whitespace. Its language constructs and object-oriented approach aim to help programmers write clear, logical code for small and large-scale projects. Python is dynamically typed and garbage collected.


**Anaconda**

Anaconda is a free and open-source distribution of the Python and R programming languages for scientific computing (data science, machine learning applications, large-scale data processing, predictive analytics, etc.), that aims to simplify package management and deployment.


## Methodology
The approach used to extract, classify, and predict features for the plant disease model is strengthened by deep learning. The model is made up of the following supporting elements:


### Analysis of Data
In order to make informed judgements, data analysis is a strategy for working with data to draw out useful information from it. When we can deduce meaning from facts, we are better able to make decisions. We also have access to a wealth of data today, so it makes sense to review it constantly given that we live in a period where we have this kind of access. We had a good idea of the picture intensity and data distribution from the data that we looked at.

![image](https://user-images.githubusercontent.com/96066261/219602270-336a9fc3-75b0-4e52-b454-f0f634235c7f.png)  ![image](https://user-images.githubusercontent.com/96066261/219602294-24e4a945-7edd-4f3c-b551-7ac5f637fc21.png)  ![image](https://user-images.githubusercontent.com/96066261/219602331-b0698495-533f-4eb1-9670-ba791eff7bbf.png)  ![image](https://user-images.githubusercontent.com/96066261/219602351-67fbd8d8-d333-439d-8783-9203b6db8aa0.png)




### Data preprocessing
The data mining approach of "data pre-processing" transforms the unprocessed data into a usable form, a format that is both useful and powerful. Since pre-processing is one of the key elements affecting the performance of the model, it is a multi-step process. We found after data analysis that the data may not be enough to train or test a suitable model. This might also be taken to suggest that deep learning models frequently outperform other models when given a lot of data. Yet it's not always possible to provide the deep learning network with a huge dataset. We used image augmentation to get around this problem. A deep learning model would not be able to successfully learn patterns from the data if there wasn't a large enough amount of data available, which was the main justification for the picture augmentation.
Image augmentation produces fresh data that can be utilised for model training by altering the existing data. We didn't use all of the augmentation techniques, but we gave the information a lot of thought before choosing.
We then changed the colour photo to black and white using picture decolorization. It is referred to as the procedure to convert a colour image to a grayscale image and is frequently used in single-channel image processing, black and white printing, etc. We chose this approach because it makes it simpler to distinguish between diseases like apple scab and apple rot that are brought on by bacterial or fungal infection. The model gains the benefit of being able to train more successfully as a result.
Also, we used Label Encoder to label the target, or the damaged plant leaves. Due to the fact that the data we were able to get was in string format, which made it challenging to train and construct the model, we separated each class of illness into integers such as 0,1,2,3. In order to overcome the obstacle, Label Encoder was used.

![image](https://user-images.githubusercontent.com/96066261/219602570-4984ca25-7dd4-4897-be76-accbe3765e94.png)


![image](https://user-images.githubusercontent.com/96066261/219602528-72ad0413-5e60-4e0c-bb0f-eec447a9954d.png)  ![image](https://user-images.githubusercontent.com/96066261/219602677-a96be10d-acd2-4d34-a0cf-72558a08f926.png)



### Model selection and training
It is simple to fit numerous machine learning models on a given predictive modelling dataset because to the availability of several libraries that offer user-friendly machine learning frameworks like scikit-learn and keras. Making the choice of model to use for a particular task is the challenge of applied machine learning. While choosing a model, it's important to take complexity, maintainability, and available resources into account in addition to performance.
We used Convolutional Neural Networks (CNN), a deep learning technique, for our model. The main motivation behind doing so was to maximise performance by applying the patterns present in the photos.
CNNs are a class of deep neural networks that are frequently employed to analyse visual data. The bottom line is that ConvNet's goal is to keep elements that are essential for making precise predictions while compressing the images into a format that is simpler to comprehend. The pooling layer is used to shrink the spatial size of the convolutional feature after the convolutional layer. The amount of computing power needed to process the data is decreased by reducing its size. Max pooling was implemented. So, what we did was determine which pixel in a certain area of the image had the highest value. This does de-noising, dimensionality reduction, and the removal of all noisy activations. The convolutional and pooling layers are employed in the model.

![image](https://user-images.githubusercontent.com/96066261/219603085-3feb34aa-b27a-4f37-93d4-af0b65a49a66.png)

![image](https://user-images.githubusercontent.com/96066261/219603134-e0995c67-5ee0-4d3e-af74-daf6928db666.png)


### Testing
Testing is the process of assessing a fully trained model's performance on a testing set. Samples that were separated from the training and validation sets make up the testing set. At this point, our models start to produce reliable forecasts. The importance of testing a model comes from the fact that many models may demonstrate excellent accuracy during the training phase but underperform when faced with unknowable input. Hence, by putting our model to the test, we can determine its limitations and strengths, which will help us improve it. In our case, we initially launched a model without any data augmentation, and the outcomes weren't good. This highlights how poorly the model performs when there aren't enough data. 
