# Real-Estate-Southern-Spain-2024
Machine-learning project that predicts property prices in Southern Spain.

## Project Structure

```
Real-Estate-Southern-Spain-2024/
├── bin/                    # Executable files
│   └── main.py            # Main Streamlit application
├── data/                   # Data files
│   └── properties.csv     # Property dataset
├── models/                 # ML preprocessing
│   └── preprocessing.ipynb # Jupyter notebook for preprocessing
├── utils/                  # Python utility functions
│   └── __init__.py
├── requirements.txt        # Python dependencies
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/LinasD22/Real-Estate-Southern-Spain-2024.git
cd Real-Estate-Southern-Spain-2024
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
3. For mac with env:

- create venv (once)
```bash
python3 -m venv .venv
```
- activate it
```bash
source .venv/bin/activate
```
- install packages listed in requirements.txt
```bash
python -m pip install -U pip
python -m pip install -r requirements.txt
```
## Usage
run the script:
```bash
 python -m streamlit run bin/main.py
```
### Running the Streamlit App

To run the main application:

```bash
streamlit run bin/main.py
```

The app will open in your default web browser. It allows you to:
- View dataset overview and statistics
- Select individual properties from a dropdown menu
- View detailed information about each property
- Browse the full dataset

### ML Preprocessing

Open the Jupyter notebook for machine learning preprocessing:

```bash
jupyter notebook models/preprocessing.ipynb
```

## Dataset

The `properties.csv` file contains property listings in Southern Spain with the following fields:
- **id**: Unique property identifier
- **location**: City/town location
- **property_type**: Type of property (Villa, Apartment, Townhouse, etc.)
- **bedrooms**: Number of bedrooms
- **bathrooms**: Number of bathrooms
- **size_sqm**: Size in square meters
- **price_eur**: Price in Euros
- **year_built**: Year of construction
- **pool**: Swimming pool availability (Yes/No)
- **garage**: Garage availability (Yes/No)
- **distance_to_beach_km**: Distance to beach in kilometers
