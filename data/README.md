# Data Directory Documentation

## 📁 Directory Structure

```
data/
├── raw/                 # Raw data files
│   ├── u.data          # User ratings
│   ├── u.item          # Movie information
│   ├── u.user          # User demographics
│   └── u.genre         # Genre information
└── processed/          # Processed data files
    ├── ratings.pt      # Processed ratings
    ├── movies.pt       # Processed movie data
    ├── users.pt        # Processed user data
    ├── features.pt     # Engineered features
    ├── graph.pt        # User-movie graph
    └── embeddings.pt   # Node embeddings
```

## 📊 Data Files

### Raw Data
1. **u.data**
   - Format: Tab-separated
   - Columns: user_id | item_id | rating | timestamp
   - Size: 100,000 ratings
   - Description: Main ratings dataset

2. **u.item**
   - Format: Tab-separated
   - Columns: movie_id | title | release_date | video_release_date | IMDb_URL | genres
   - Size: 1,682 movies
   - Description: Movie metadata and genre information

3. **u.user**
   - Format: Tab-separated
   - Columns: user_id | age | gender | occupation | zip_code
   - Size: 943 users
   - Description: User demographic information

4. **u.genre**
   - Format: Text file
   - Content: List of 19 movie genres
   - Description: Genre definitions

### Processed Data
1. **ratings.pt**
   - Format: PyTorch tensor
   - Shape: [100000, 4]
   - Content: Processed ratings with time decay

2. **movies.pt**
   - Format: PyTorch tensor
   - Shape: [1682, 25]
   - Content: Movie features and genre vectors

3. **users.pt**
   - Format: PyTorch tensor
   - Shape: [943, 8]
   - Content: User features and demographics

4. **features.pt**
   - Format: PyTorch tensor
   - Shape: [100000, 32]
   - Content: Engineered features for model input

5. **graph.pt**
   - Format: PyTorch Geometric Data object
   - Content: User-movie interaction graph
   - Properties: Node features, edge indices, edge weights

6. **embeddings.pt**
   - Format: PyTorch tensor
   - Shape: [2625, 64]
   - Content: Node embeddings for users and movies

## 🔄 Data Processing Pipeline

1. **Data Loading**
   - Load raw files
   - Handle missing values
   - Convert data types

2. **Preprocessing**
   - Normalize ratings
   - Encode categorical features
   - Process timestamps

3. **Feature Engineering**
   - Create time decay features
   - Generate genre vectors
   - Build user preference profiles
   - Create quantum embeddings

4. **Graph Construction**
   - Build user-movie graph
   - Generate node embeddings
   - Create edge features

## 📝 Usage

```python
from src.data.processor import DataProcessor

# Initialize processor
processor = DataProcessor()

# Load and process data
processor.load_data()
processor.preprocess_data()
processor.create_features()
processor.save_processed_data()
```

## 🔍 Data Quality Checks

1. **Completeness**
   - No missing ratings
   - All users have demographics
   - All movies have genre information

2. **Consistency**
   - Rating scale: 1-5
   - Timestamp format: Unix seconds
   - Genre encoding: Binary vectors

3. **Distribution**
   - Rating distribution analysis
   - User activity patterns
   - Genre popularity

## 📚 References

- [MovieLens Dataset Documentation](https://grouplens.org/datasets/movielens/)
- [Data Processing Code](src/data/processor.py)
- [Graph Construction Code](src/data/graph.py) 