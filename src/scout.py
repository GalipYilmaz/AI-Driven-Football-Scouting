import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

class ScoutEngine:
    def __init__(self, data_path="data/cleaned_players.csv"):
        # Load the cleaned dataset
        self.df = pd.read_csv(data_path)

        # Features used for KNN similarity
        self.features = [
            'norm_overall', 'norm_potential', 'norm_pace',
            'norm_shooting', 'norm_passing', 'norm_dribbling',
            'norm_defending', 'norm_physic'
        ]

        self.scaler = StandardScaler()
        self.model = NearestNeighbors(metric='euclidean')
        self._train_model()

    def _train_model(self):
        # Prepare scaled feature matrix
        X = self.df[self.features]
        self.X_scaled = self.scaler.fit_transform(X)

    def find_similar_players(self, player_name, n_players=5, max_price=None, max_age=None, league=None):
        player_query = self.df[self.df['short_name'] == player_name]
        if player_query.empty:
            return f"Error: Player '{player_name}' not found."

        player_index = player_query.index[0]
        player_features = self.df.loc[[player_index], self.features]
        player_scaled = self.scaler.transform(player_features)

        # Candidate pool excluding target
        pool_df = self.df.drop(index=player_index).copy()

        # PRE-FILTERING
        if max_price is not None:
            pool_df = pool_df[pool_df['value_eur'] <= max_price]
        if max_age is not None:
            pool_df = pool_df[pool_df['age'] <= max_age]
        if league is not None:
            pool_df = pool_df[pool_df['league_name'] == league]

        if pool_df.empty:
            return "No players found matching your strict filters."

        pool_indices = pool_df.index
        pool_scaled = self.X_scaled[pool_indices]

        # Temporary KNN on filtered subset
        dynamic_n = min(n_players, len(pool_df))
        temp_model = NearestNeighbors(n_neighbors=dynamic_n, metric='euclidean')
        temp_model.fit(pool_scaled)

        distances, indices = temp_model.kneighbors(player_scaled)
        similar_indices = pool_indices[indices[0]]
        similar_distances = distances[0]

        # RESULTS: Added 'player_positions' to avoid KeyError in app.py
        results = self.df.loc[similar_indices][
            ['player_id', 'short_name', 'player_positions', 'age', 'overall', 'potential',
             'value_eur', 'wage_eur', 'club_name', 'league_name', 'player_url']
        ].copy()

        results["distance_score"] = similar_distances.round(3)
        return results