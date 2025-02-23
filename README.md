# Steam Game Recommendation System
## Project Overview
This project aims to build a personalized game recommendation system based on game data from the Steam platform. It leverages various machine learning techniques to generate precise game recommendations for users while exploring a combination of classic collaborative filtering and deep learning experiments. Implemented in Python, the project primarily uses the Surprise library to build the collaborative filtering model and integrates experiments with TensorFlow/Keras for deep learning, helping players discover games they might enjoy and providing insights into user preferences and game trends for data analysts.

## Data Source
The project uses data from the Game Recommendations on Steam dataset available on Kaggle(https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam/data). This dataset contains extensive user ratings and game attribute information, providing a solid foundation for model training.

## How to Use
The project mainly relies on the following Python libraries for data processing, model building, and evaluation:

- pandas and numpy: For data loading and numerical computations.
- json: For configuration or data parsing (if needed).
- TensorFlow and Keras: For building and experimenting with deep learning models to explore additional recommendation approaches.
- Surprise: Primarily used to build the collaborative filtering recommendation model, which employs the SVD algorithm and an item-based collaborative filtering approach. It also uses Surprise’s Dataset, Reader, cross_validate, and accuracy modules for model training and evaluation.
- Kria: Assists in model tuning and hyperparameter search.

## Setup
git clone https://github.com/JiayiWang78/Game-Recommendations-on-Steam.git cd Game-Recommendations-on-Steam

Open the steamgames (1).ipynb file and execute the code cells sequentially to view the complete process of data preprocessing, model building, tuning, and evaluation.

## Recommendation Algorithm Description
The recommendation system is divided into the following steps:

### (1)Data Preprocessing:

Perform simple handling of NA values in the original dataset.
Merge data from three different sources and filter to select the top 5000 games based on popularity, ensuring data quality and improved recommendation performance.
### (2)Collaborative Filtering Model (Based on Surprise):

Use the Dataset and Reader modules from the Surprise library to construct the user-game rating dataset.
Item-Based Collaborative Filtering: Set up the following configuration to use cosine similarity for calculating similarities between games:

sim_options = {
    "name": "cosine",  # cosine similarity
    "user_based": False  # item (game) recommendation
}

- Here, user_based=False indicates that an item-centric recommendation approach is used, recommending games similar to those in the user’s history.
- Apply the SVD algorithm to extract latent features between users and games and predict ratings.
- Perform cross-validation using cross_validate and evaluate the model using RMSE (Root Mean Square Error) from Surprise’s accuracy module.
### (3)Deep Learning Experiments:

- Import TensorFlow and Keras to attempt building neural network models (e.g., using embedding layers) to explore deep learning-based recommendation approaches as a comparative experiment.
- Compare the prediction accuracy of traditional collaborative filtering models with that of deep learning models.

## Evaluation Method
The project mainly uses Root Mean Square Error (RMSE) as the evaluation metric. RMSE is calculated through cross-validation provided by Surprise to measure the error between predicted and actual ratings, thereby assessing the recommendation model's performance.

## Project Structure
├── steamgames (1).ipynb

├── recommendations.csv

├── games_metadata.json

├── users.csv

├── games.csv

└── README.md

Contributions are welcome! If you have any questions or suggestions for improvement, please submit an issue or a pull request.
