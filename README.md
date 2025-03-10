# FPL Transfer Recommender

A machine learning-based system for recommending optimal Fantasy Premier League (FPL) player transfers based on predicted points, team constraints, and budget considerations.

## Overview

The Fantasy Premier League Transfer Recommender is a comprehensive tool designed to help FPL managers make data-driven decisions when selecting transfers for their teams. The system leverages machine learning algorithms to predict player performance and provide optimized transfer recommendations that maximize expected points while respecting team structure and budget constraints.

### Key Features

- **Data Integration**: Seamless integration with the FPL API to fetch up-to-date player statistics, fixtures, and team data
- **Feature Engineering**: Advanced feature creation based on form, fixture difficulty, team strength, and positional metrics
- **Machine Learning Models**: Predicts future player performance using various ML algorithms (Random Forest, Gradient Boosting, etc.)
- **Transfer Optimization**: Recommends optimal transfers considering team constraints, budget, and expected points
- **Chip Strategy**: Provides advice on when to use chips like Wildcard, Free Hit, etc.
- **Command Line Interface**: Easy-to-use interface for generating recommendations

## Installation

### Prerequisites

- Python 3.8+
- pip
- Poetry (optional but recommended for dependency management)

### Installation Steps

1. Clone the repository:

```bash
git clone https://github.com/yourusername/fpl-transfer-recommender.git
cd fpl-transfer-recommender
```

2. Install dependencies using Poetry:

```bash
poetry install
```

Or using pip:

```bash
pip install -r requirements.txt
```

3. Set up the environment:

```bash
cp .env.example .env
# Edit .env with your FPL credentials if needed
```

## Usage

### Command Line Interface

The simplest way to use the system is through the command-line interface:

```bash
python -m fpl_transfer_recommender.main --user_id=YOUR_FPL_ID --transfers=1 --budget=0.5
```

Arguments:
- `--user_id`: Your FPL user ID (required)
- `--transfers`: Number of transfers you want to make (default: 1)
- `--budget`: Additional budget in millions (default: 0.0)

### Python API Usage

You can also use the package as a library in your own Python code:

```python
from fpl_transfer_recommender.data.api import get_player_data, get_team_data, get_fixture_data, get_user_team_data
from fpl_transfer_recommender.data.preprocessing import get_preprocessed_data
from fpl_transfer_recommender.models.feature_engineering import generate_full_feature_set
from fpl_transfer_recommender.models.recommendation import TransferRecommender

# Fetch data
player_data = get_player_data()
team_data = get_team_data()
fixture_data = get_fixture_data()

# Preprocess data
processed_data = get_preprocessed_data(player_data, team_data, fixture_data)

# Generate features
players_with_features = generate_full_feature_set(
    processed_data['players'], 
    processed_data['fixtures'], 
    processed_data['teams']
)

# Get user team
user_id = 12345  # Replace with your FPL ID
user_team = get_user_team_data(user_id)

# Create recommender
recommender = TransferRecommender()

# Get recommendations
recommendations = recommender.optimize_transfers(
    user_team, 
    players_with_features, 
    budget=0.5, 
    num_transfers=1
)

print(recommendations)
```

## Project Structure

The project follows a modular architecture:

```
fpl_transfer_recommender/
├── __init__.py
├── main.py                   # Main entry point and CLI
├── data/
│   ├── __init__.py
│   ├── api.py                # FPL API interaction
│   └── preprocessing.py      # Data cleaning and preparation
├── models/
│   ├── __init__.py
│   ├── feature_engineering.py # Feature creation
│   ├── training.py           # ML model training
│   └── recommendation.py     # Transfer recommendations
└── utils/
    ├── __init__.py
    ├── config.py             # Configuration settings
    └── helpers.py            # Utility functions
```

## Machine Learning Approach

The system uses supervised learning to predict player performance. The main steps include:

1. **Data Collection**: Fetching historical player statistics, fixtures, and team data
2. **Feature Engineering**: Creating predictive features based on player form, fixture difficulty, etc.
3. **Model Training**: Training regression models to predict player points
4. **Transfer Optimization**: Using predicted points to recommend optimal transfers

We support multiple machine learning models including Random Forest, Gradient Boosting, and linear models, with hyperparameter tuning capabilities.

## Transfer Recommendation Logic

The recommendation engine considers:

- Predicted points for each player
- Current team composition
- Transfer budget
- Team constraints (max 3 players from same team)
- Position requirements (2 GKP, 5 DEF, 5 MID, 3 FWD)

It employs optimization strategies to find the best combination of transfers that maximize expected points gain.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [amosbastian/fpl](https://github.com/amosbastian/fpl) - Python wrapper for the FPL API
- Fantasy Premier League for providing the data
- The FPL community for inspiration and feedback

## Contact

If you have any questions or suggestions, feel free to open an issue or contact the maintainers.

---

Happy FPL managing! May your arrows be green and your rank high!

