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

        # We will not set a static n_neighbors here anymore,
        # because the pool size will change dynamically with filters.
        # We only define the metric.
        self.model = NearestNeighbors(metric='euclidean')

        self._train_model()

    def _train_model(self):
        # Train the base KNN model (useful if we want a global search later)
        X = self.df[self.features]

        # Scale the features
        X_scaled = self.scaler.fit_transform(X)

        # Store the scaled features for fast slicing during pre-filtering
        self.X_scaled = X_scaled

    def find_similar_players(self, player_name, n_players=5, max_price=None, max_age=None, league=None):
        # Find the index of the player in the DataFrame
        player_index = self.df[self.df['short_name'] == player_name].index

        # Check if the player is found
        if player_index.empty:
            return f"Error: Player '{player_name}' not found in the dataset."

        player_index = player_index[0]

        # Get the features of the target player and scale them
        player_features = self.df.loc[[player_index], self.features]
        player_scaled = self.scaler.transform(player_features)

        # Create a pool excluding the target player
        pool_df = self.df.drop(index=player_index).copy()

        # --- PRE-FILTERING ---
        if max_price is not None:
            pool_df = pool_df[pool_df['value_eur'] <= max_price]
        if max_age is not None:
            pool_df = pool_df[pool_df['age'] <= max_age]
        if league is not None:
            pool_df = pool_df[pool_df['league_name'] == league]

        # Handle edge case: If filters are too strict and pool is empty
        if pool_df.empty:
            return "No players found matching your strict filters."

        # Get the pre-scaled features only for the filtered pool
        pool_indices = pool_df.index
        pool_scaled = self.X_scaled[pool_indices]

        # Train a temporary model strictly on the filtered pool
        # Make sure we don't ask for more neighbors than the pool size
        dynamic_n_neighbors = min(n_players, len(pool_df))
        temp_model = NearestNeighbors(n_neighbors=dynamic_n_neighbors, metric='euclidean')
        temp_model.fit(pool_scaled)

        # Find the nearest neighbors within this specific pool
        distances, indices = temp_model.kneighbors(player_scaled)

        # Map the local pool indices back to the global DataFrame indices
        similar_indices = pool_indices[indices[0]]
        similar_distances = distances[0]

        # Create a new DataFrame with the results
        results = self.df.loc[similar_indices][
            ['short_name', 'age', 'overall', 'potential', 'value_eur',
             'wage_eur', 'club_name', 'player_url', 'team_url']
        ].copy()

        # Add the 'distance_score' column
        results["distance_score"] = similar_distances.round(3)

        return results


# --- TEST ---
if __name__ == "__main__":
    # Initialize the engine
    engine = ScoutEngine()

    # Set the target player
    target_player = "Baris Alper Yilmaz"

    print(f"\n--- Target Player ({target_player}) Stats ---")
    target_stats = engine.df[engine.df['short_name'] == target_player][
        ['short_name', 'age', 'overall', 'potential', 'value_eur', 'wage_eur', 'league_name']
    ]
    print(target_stats.to_string(index=False))

    # Set the filters
    budget = 15000000.0
    age_limit = 24

    print(f"\n--- Scouting young and cheap alternatives to {target_player} ---")
    print(f"Filters: Max Price = €{budget / 1000000}M | Max Age = {age_limit}")

    # Find similar players and print the results
    similar_players = engine.find_similar_players(
        player_name=target_player,
        max_price=budget,
        max_age=age_limit
    )

    # Print handling for string errors vs dataframe results
    if isinstance(similar_players, str):
        print(similar_players)
    else:
        print(similar_players.to_string(index=False))