import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, Union, List
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError


class DelayModel:

    FEATURES_COLS = [
        "OPERA_Latin American Wings",
        "MES_7",
        "MES_10",
        "OPERA_Grupo LATAM",
        "MES_12",
        "TIPOVUELO_I",
        "MES_4",
        "MES_11",
        "OPERA_Sky Airline",
        "OPERA_Copa Air"
    ]

    def __init__(self):
        self._model = LogisticRegression(class_weight="balanced", random_state=42)

    def _get_min_diff(self, row: pd.Series) -> float:
        fecha_o = datetime.strptime(row["Fecha-O"], "%Y-%m-%d %H:%M:%S")
        fecha_i = datetime.strptime(row["Fecha-I"], "%Y-%m-%d %H:%M:%S")
        min_diff = (fecha_o - fecha_i).total_seconds() / 60
        return min_diff

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or prediction.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        
        features = pd.concat([
            pd.get_dummies(data["OPERA"], prefix="OPERA"),
            pd.get_dummies(data["TIPOVUELO"], prefix="TIPOVUELO"),
            pd.get_dummies(data["MES"], prefix="MES")
        ], axis=1)

        #Verificamos que las 10 columnas requeridas existan
        for col in self.FEATURES_COLS:
            if col not in features.columns:
                features[col] = 0

        #Se filtra para verificar que las 10 columnas seleccionadas estén en el orden  correcto
        features = features[self.FEATURES_COLS]

        if target_column is not None:
            if target_column not in data.columns:
                min_diffs = data.apply(self._get_min_diff, axis=1)
                threshold_in_minutes = 15
                target = np.where(min_diffs > threshold_in_minutes, 1, 0)
                target = pd.DataFrame({target_column: target}, index=data.index)
            else:
                target = data[[target_column]]

            return features, target

        return features

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        y = target.values.ravel() if isinstance(target, pd.DataFrame) else target
        self._model.fit(features, y)

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            (List[int]): predicted targets.
        """
        if self._model is None:
            return [0] * len(features)

        #Garantía que las características vengan ordenadas
        for col in self.FEATURES_COLS:
            if col not in features.columns:
                features[col] = 0

        features_filtered = features[self.FEATURES_COLS]

        #Checamos el caso en que el modelo aún no se haya ajustado
        try:
            predictions = self._model.predict(features_filtered)
        except NotFittedError:
            return [0] * len(features_filtered)

        return [int(pred) for pred in predictions]