# Flipkart_Data_Classification_using_Description
Using Flipkart products data, building a model to predict the category using description.

## Steps to Setup

Step 1 : Clone the Repo :  ```git clone https://github.com/manju1201/Flipkart_Data_Classification_using_Description.git```

## About files:

### Data files
* Given data - [flipkart_com-ecommerce_sample - flipkart_com-ecommerce_sample - flipkart_com-ecommerce_sample](https://drive.google.com/file/d/1lwfF_6Mve5lA_DmtrE35Yg1dkEpoH5Jj/view?usp=sharing) 
* After data Analysis - [1_Post_Data_Analysis_flipkart_com-ecommerce_sample](https://drive.google.com/file/d/1_hli8uUtPfBytvGYiuI6X724M9_sBF-v/view?usp=sharing)
* After Cleaning Data - [2_cleaned_flipkart_com_ecommerce_sample](https://drive.google.com/file/d/1-HitZ9cZc8CIylLt2IQkkfiiygISNsTY/view?usp=sharing)
* Data for CNN model - [decscription](https://drive.google.com/file/d/1-9mes59R-7A8mpAnRc1d1B1e6RXlmZm0/view?usp=sharing) and
                        [Primary Category](https://drive.google.com/file/d/1-ILU_evhgFLS9MeqqrnpU9SPpsgja60p/view?usp=sharing)


### Ipython Notebooks:
* 1_Data_Analysis_&_Visualization.ipynb 
* 2_Cleaning_Data.ipynb
* 3_Product_Classification_Using_ML_Models.ipynb
* 4_Product_Classification_Using_CNN_Model.ipynb

### Weights file:
* Model-conv1d.h5

# The basic flow followed here is
* Data Analysis
* Data Visualization
* Cleaning and Preprocessing
* Finding and Cleaning the Primary category label
* Countvectorizer
* Models
* Results
* Improvements
* Resources

## Data Analysis
* Dataset named “flipkart_com-ecommerce_sample - flipkart_com-ecommerce_sample - flipkart_com-ecommerce_sample.csv” contains about 20000 rows and 15 columns/features.
* Columns in the dataset.

![alt text](https://github.com/manju1201/Flipkart_Data_Classification_using_Description/blob/main/Images/coulmns.png)
* Some of the columns have missing values, they are: ‘retail_price’, ‘discounted_price’, ‘image’, ‘description’, ‘brand’ and ‘product_specifications’

![alt_text](https://github.com/manju1201/Flipkart_Data_Classification_using_Description/blob/main/Images/null_count.png)
* Could not find any missing values in ‘product_rating’ and ‘overall_rating’ but the  value is "No rating available" for lot of rows

![alt_text](https://github.com/manju1201/Flipkart_Data_Classification_using_Description/blob/main/Images/product_rating.png)
![alt_text](https://github.com/manju1201/Flipkart_Data_Classification_using_Description/blob/main/Images/overall_rating.png)
* Splitted the category_level_tree into 6 levels of categories. 
![alt_text](https://github.com/manju1201/Flipkart_Data_Classification_using_Description/blob/main/Images/required_data.png)
* Lets see the “Unique” and “None” values in each of the category level

![alt_text](https://github.com/manju1201/Flipkart_Data_Classification_using_Description/blob/main/Images/unique_values.png) ![alt_text](https://github.com/manju1201/Flipkart_Data_Classification_using_Description/blob/main/Images/None_values.png)
* In Category Level 1, most of the data is in Clothing(6198), Jewellery(3531) and Footwear(1227) and there are no “None” Values.
![alt_text](https://github.com/manju1201/Flipkart_Data_Classification_using_Description/blob/main/Images/cat_level_1_graph.png)
* In category level 2, here also the majority data is in Clothing like women’s clothing(3901) and Men’s Clothing(1773) and we can see about 328 “None” values.
![alt_text](https://github.com/manju1201/Flipkart_Data_Classification_using_Description/blob/main/Images/cat_level_2_graph.png)
* In Category level 3 also Clothing that is Western wear(1981) is in the top but “None” values in this level is too high that is about 1457.
![alt_text](https://github.com/manju1201/Flipkart_Data_Classification_using_Description/blob/main/Images/cat_level_3_graph.png)
* In Category level 4, the top result itself is “None”, about 5876 null values. So we can conclude that this cant be our primary category for prediction.
![alt_text](https://github.com/manju1201/Flipkart_Data_Classification_using_Description/blob/main/Images/cat_level_4_graph.png)
* In category levels 5 and 6 also has about 10642 and 15552 respectively.

## Cleaning and Preprocessing Data
* All the examples were converted into strings and underwent processing.
* Every character except alphabets are replaced with space.
* Multiple spaces are replaced with single space.
* Converted the cleaned string to lowercase which is called normalization.
* Now, the words are tokenized and the stop words and words of length less than 2  are removed.
* If we have two words like ‘see’ and ‘seeing’ then they would be stored in the index as two different terms whereas in actual sense we wouldn’t need them as two different words. So we take into consideration millions of documents in real time. This would create a huge amount of memory and take huge process time. So, the best way to index only necessary terms is to stem them. So, for the above example, after lemmatization the terms stored in the index is just ‘see’.
* All the preprocessing is done and the cleaned tokens are combined back into sentences.
* There are about 2 null values in description, as description is our main feature I dropped those two empty rows.

## Finding and Cleaning the Primary category label
* We have about 266 labels and 0 None values in Category level 1 whereas 224 labels and 328 None values in Category level 2
* As there are no None values in Category level 1 , this is considered as the Primary Category for the further implementation.
![alt_text](https://github.com/manju1201/Flipkart_Data_Classification_using_Description/blob/main/Images/cat_level_1_graph.png)
* Looking at the plot we can conclude the data is not equally distributed and most of them are having subcategory value 1. 
* First I thought to drop the subcategory rows which have less than 15 values.
* As it is mentioned in the problem that data can be cleaned and processed therefore I did some manual work so that labels are reduced from 266 to 27.
* For this I looked at each of the labels and searched the product in flipkart website to find the primary category and created about 23 lists where the lists contain keywords from the labels which have less than 15 values.
* If the keyword is present in a label then that label is renamed to the keyword label list. After this process, we end up with 27 labels. 
* One row was dropped, as its primary label could not be found due to lack of data.
* Data is prepared, the model is built on 19997 rows/examples with 27 labels.

# Tokenizer
#### Countvectorizer
* The CountVectorizer provides a simple way to both tokenize a collection of text documents and build a vocabulary of known words, but also to encode new documents using that vocabulary.

# Models

* Dataset containing 19997 examples is split into Train Set 80% and Test Set 20%.

#### Multinomial Naive Bayes
* Multinomial Naïve Bayes uses term frequency i.e. the number of times a given term appears in a document. After normalization, term frequency can be used to compute maximum likelihood estimates based on the training data to estimate the conditional probability.

#### Decision Tree Classifier
* The decision tree classifier creates the classification model by building a decision tree. Each node in the tree specifies a test on an attribute, each branch descending from that node corresponds to one of the possible values for that attribute.
* Parameter used is random state, this parameter controls the randomness of the estimator. To obtain a deterministic behaviour during fitting, random_state has to be fixed to an integer.

### Random Forest Classifier
* Random forests are an ensemble learning method and a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy, control over-fitting and outputs the class that is the mode of the classes of the individual trees.
* max_depth represents the depth of each tree in the forest. The deeper the tree, the more splits it has and it captures more information about the data. 
Controls sampling of the features to consider when looking for the best split at each node


#### CNN Model
* To improve prediction I have also tried CNN model
* After all the preprocessing I have also found class_weights 
    * Data is not equally distributed so assigned higher weights to lower classes and lower weights to higher classes.

##### Hyperparameters used
* Epoch = 20
* Bactch_size = 32

* This CNN model is built with 6 layers
    * Embedding layer - Embedding layer enables us to convert each word into a fixed length vector of defined size. The resultant vector is a dense one with real values instead of just 0's and 1's. The fixed length of word vectors helps us to represent words in a better way along with reduced dimensions.
    
    * Dropout layer - The Dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting. Here the rate is 0.1.
    
    * Conv1D layer - This layer creates a convolution kernel that is convolved with the layer input over a single spatial dimension to produce a tensor of outputs. Here stride-1, valid padding, filter size of 300 is used.
    
    * Global Max Pool 1D - Downsamples the input representation by taking the maximum value over the time dimension.
    
    * Dense Layer -  This is a densely-connected Neural Network layer.
    
    * Activation - Applying an sigmoid activation function to an output.

* Adam Optimizer - Adam optimization is a stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments.

* Binary cross entropy loss function - Computes the cross-entropy loss between true labels and predicted labels.

* Categorical accuracy - Calculates how often predictions match with one-hot labels.

## Results

##### Multinomial naive bayes
  * Accuracy - 93.8%
  * Weighted F1 score - 93%

##### Decision tree Classifier
  * Accuracy - 95.6%
  * Weighted F1 score - 96% 

##### Random forest classifier
  * Accuracy - 96.8%
  * Weighted F1 score - 97% 

#### CNN
  * Accuracy - 97.0%

### Improvements
* Creating word vectors by training word2vec or fasttext models on a huge products corpus and using them to initialize the cnn model for training it. 
* State of art models like transformer based models can be used.


### Resources
* [Bar Charts and Heatmaps](https://www.kaggle.com/alexisbcook/bar-charts-and-heatmaps)
* [Data Cleaning](https://www.kaggle.com/learn/data-cleaning)
* [Basic Text Classification](https://www.kaggle.com/matleonard/text-classification)
* [CNN Model Refernce](https://realpython.com/python-keras-text-classification/#choosing-a-data-set)
