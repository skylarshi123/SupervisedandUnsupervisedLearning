# SupervisedandUnsupervisedLearning
 I applied Clustering for personality tests and supervised learning to predict demand of passengers basd on street location
# Personality Test Analysis and Demand Prediction

This project focuses on two main tasks: analyzing a personality test dataset and predicting the demand for taxi services based on location, time, and date. The project utilizes various machine learning techniques, including clustering, dimensionality reduction, and regression.

## Dataset 1: Personality Test Analysis

The first dataset, `personality_test.csv`, contains the answers of various individuals to 50 different questions related to personality traits. The questions are categorized into five dimensions: extroversion, neuroticism, agreeableness, conscientiousness, and openness.

### Clustering and Dimensionality Reduction

To analyze the personality test data, we applied clustering algorithms to group individuals based on similar features. Specifically, we used the Agglomerative Clustering algorithm and determined the optimal number of clusters using the elbow method.

To visualize the clusters, we applied Principal Component Analysis (PCA) to reduce the dimensionality of the dataset. This allowed us to plot the data points in a 2D space, making it easier to observe the clusters.

### Personality Analysis

In addition to clustering, we developed a `QuestionnaireAnalyzer` class that takes an individual's question scores as input and calculates the average score for each personality dimension. This enables us to provide personalized insights into an individual's personality based on their responses to the questionnaire.

## Dataset 2: Demand Prediction

The second dataset, `Demand.csv`, contains information about the ordering of taxis based on location, time, and date. The goal is to build an accurate predictive model that helps the company estimate the demand for taxis at specific locations and times.

### Data Preprocessing

Before building the predictive model, we performed data preprocessing steps, including encoding categorical variables using the Label Encoder from scikit-learn.

### Model Selection and Tuning

We utilized the PyCaret library to compare various regression models and select the best-performing one. The Light Gradient Boosting Machine (LightGBM) model emerged as the top choice based on evaluation metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared.

To further improve the model's performance, we performed hyperparameter tuning using PyCaret's `tune_model` function. This allowed us to find the optimal set of hyperparameters for the LightGBM model.

### Outlier Detection

To handle potential outliers in the demand data, we employed the Isolation Forest algorithm. By setting a contamination threshold, we identified and flagged outliers in the dataset. This step helps to ensure the robustness of the predictive model.

## Getting Started

To run the code and reproduce the results, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/your-username/personality-test-analysis-demand-prediction.git
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebooks in the following order:
   - `personality_test_analysis.ipynb`: Performs clustering and personality analysis on the personality test dataset.
   - `demand_prediction.ipynb`: Builds and tunes the predictive model for taxi demand based on location, time, and date.

4. Explore the notebooks, modify the code as needed, and experiment with different algorithms and parameters.

## Conclusion

This project demonstrates the application of machine learning techniques for personality analysis and demand prediction. By leveraging clustering, dimensionality reduction, and regression algorithms, we gained insights into personality traits and developed an accurate model to forecast taxi demand.

The code is well-documented and provides a solid foundation for further exploration and refinement. Feel free to experiment with different algorithms, tune hyperparameters, and extend the functionality to suit your specific requirements.

We hope this project serves as a valuable resource for anyone interested in personality analysis and demand prediction using machine learning.
