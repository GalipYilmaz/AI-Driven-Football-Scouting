# ⚽ PRO-SCOUT AI | AI-Driven Football Scouting

An advanced, AI-powered football scouting tool designed to find player alternatives using Machine Learning. This tool compares players using K-Nearest Neighbors (KNN) and visualizes technical attributes with dual-radar charts.
Used EA FC24 dataset.


## 🚀 Features

* **AI Matchmaking:** Uses KNN (K-Nearest Neighbors) to find the most similar players based on normalized performance metrics.
* **Dynamic Filtering:** Filter results by **League, Position, Budget, Age,** and **Overall/Potential** ratings.
* **Dual-Radar Analysis:** Compare the target player and recommendations side-by-side in a single, overlapping radar chart.
* **Interactive Comparison:** Head-to-head bar chart analysis for specific attributes (Pace, Shooting, Passing, etc.)
* **SoFIFA Integration:** Quick access to detailed player profiles via direct links.
* **Pagination & Sorting:** Clean result management with alphabetical or similarity-based sorting.

## 🛠️ Tech Stack

* **Python 3.12**
* **Streamlit** (Web Interface)
* **Pandas** (Data Manipulation)
* **Scikit-Learn** (KNN Algorithm)
* **Plotly** (Interactive Visualizations)

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/rhymasell/AI-Driven-Football-Scouting.git](https://github.com/rhymasell/AI-Driven-Football-Scouting.git)
   cd AI-Driven-Football-Scouting

2. Create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

3. Install dependencies:
   ```bash
   pip install -r requirements.txt

4. Run the application:
   ```bash
   streamlit run src/app.py

## How It Works
The engine calculates the  Euclidean distance between players using normalized features:

* **Pace, Shooting, Passing, Dribbling, Defending, Physical**

By applying a dynamic KNN model on a pre-filtered candidate pool, the system provides real-time recommendations that fit specific tactical and budgetary constraints.

Developed by Galip Yılmaz
