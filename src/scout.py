import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

class ScoutEngine:
    def __init__(self, data_path="data/cleaned_players.csv"):
        # Load the dataset
        self.df = pd.read_csv(data_path)

        # KNN features
        self.features = [
            'norm_overall', 'norm_potential', 'norm_pace',
            'norm_shooting', 'norm_passing', 'norm_dribbling',
            'norm_defending', 'norm_physic'
        ]

        # Initialize the scaler
        self.scaler = StandardScaler()

        self.model = NearestNeighbors(n_neighbors=6, metric='euclidean')

        self._train_model()

    def _train_model(self):
        # Train the KNN model
        X = self.df[self.features]

        # Scale the features
        X_scaled = self.scaler.fit_transform(X)

        # Fit the model
        self.model.fit(X_scaled)

        # Store the scaled features
        self.X_scaled = X_scaled

    def find_similar_players(self, player_name, n_players=5):
        # Find the index of the player in the DataFrame
        player_index = self.df[self.df['short_name'] == player_name].index

        # Check if the player is found in the DataFrame
        if player_index.empty:
            print(f"Player '{player_name}' not found in the dataset.")

        player_index = player_index[0]

        # Get the features of the player
        player_features = self.df.loc[player_index, self.features].values.reshape(1, -1)
        player_scaled = self.scaler.transform(player_features)

        # Find the nearest neighbors
        distances, indices = self.model.kneighbors(player_scaled, n_neighbors=n_players+1)

        # Get the similar players except the player itself
        similar_indices = indices[0][1:]
        similar_distances = distances[0][1:]

        # Create a new DataFrame with the similar players
        results = self.df.iloc[similar_indices][
            ['short_name', 'age', 'overall', 'potential', 'value_eur', 'wage_eur', 'league_name']
        ].copy()

        # Create the 'distance_score' column
        results["distance_score"] = similar_distances.round(3)

        return results


# --- TEST ---
if __name__ == "__main__":
    # Initialize the engine
    engine = ScoutEngine()

    # Set the target player
    target_player = "L. Torreira"
    print(f"\n--- Scouting players similar to {target_player} ---")

    # Find similar players and print the results
    similar_players = engine.find_similar_players(target_player)
    print(similar_players.to_string(index=False))