import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import joblib
import os

from utils.logger import get_logger

logger = get_logger()

class CustomFeatureTransformer(BaseEstimator, TransformerMixin):
    """Classe pour la création de fonctionnalités personnalisées"""
    
    def __init__(self):
        """Initialise le transformateur de fonctionnalités"""
        pass
    
    def fit(self, X, y=None):
        """
        Ajuste le transformateur aux données (ne fait rien dans ce cas)
        
        Args:
            X (pd.DataFrame): Données d'entrée
            y: Variable cible (non utilisée)
        
        Returns:
            self: Le transformateur
        """
        return self
    
    def transform(self, X):
        """
        Transforme les données en ajoutant de nouvelles fonctionnalités
        
        Args:
            X (pd.DataFrame): Données d'entrée
        
        Returns:
            pd.DataFrame: Données avec nouvelles fonctionnalités
        """
        logger.info("Création de fonctionnalités personnalisées")
        X_copy = X.copy()
        
        # Convertir les colonnes de type 'object' en chaînes de caractères
        for col in X_copy.select_dtypes(include=['object']).columns:
            X_copy[col] = X_copy[col].astype(str)
        
        # Création de fonctionnalités liées aux services
        self._create_service_features(X_copy)
        
        # Création de fonctionnalités liées aux contrats et à la tenure
        self._create_contract_tenure_features(X_copy)
        
        # Création de fonctionnalités liées aux charges
        self._create_charge_features(X_copy)
        
        logger.info(f"Fonctionnalités créées: {X_copy.shape[1] - X.shape[1]} nouvelles fonctionnalités")
        
        return X_copy
    
    def _create_service_features(self, X):
        """
        Crée des fonctionnalités liées aux services souscrits
        
        Args:
            X (pd.DataFrame): Données d'entrée
        """
        service_columns = [
            'PhoneService', 'MultipleLines', 'InternetService', 
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        
        # Vérifier si les colonnes existent dans le DataFrame
        service_columns = [col for col in service_columns if col in X.columns]
        
        if not service_columns:
            logger.warning("Aucune colonne de service trouvée dans les données")
            return
        
        # Nombre total de services souscrits
        X['TotalServices'] = 0
        
        for col in service_columns:
            # Pour les colonnes binaires
            if set(X[col].unique()).issubset({'Yes', 'No', 'No phone service', 'No internet service'}):
                X['TotalServices'] += (X[col] == 'Yes').astype(int)
        
        # Indicateur de client avec multiple services
        X['HasMultipleServices'] = (X['TotalServices'] > 1).astype(int)
        
        # Indicateur de client premium (avec beaucoup de services)
        X['IsPremiumCustomer'] = (X['TotalServices'] >= 5).astype(int)
        
        # Caractéristiques spécifiques aux services Internet
        internet_services = [
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        
        internet_services = [col for col in internet_services if col in X.columns]
        
        if internet_services:
            X['InternetServicesCount'] = 0
            for col in internet_services:
                X['InternetServicesCount'] += (X[col] == 'Yes').astype(int)
    
    def _create_contract_tenure_features(self, X):
        """
        Crée des fonctionnalités liées aux contrats et à la durée d'abonnement
        
        Args:
            X (pd.DataFrame): Données d'entrée
        """
        # Vérifier si les colonnes nécessaires existent
        required_cols = ['tenure', 'Contract']
        missing_cols = [col for col in required_cols if col not in X.columns]
        
        if missing_cols:
            logger.warning(f"Colonnes manquantes pour les fonctionnalités contrat/tenure: {missing_cols}")
            return
        
        # Catégorisation de la tenure
        tenure_bins = [0, 12, 24, 36, 48, 60, 72, np.inf]
        tenure_labels = ['0-12', '13-24', '25-36', '37-48', '49-60', '61-72', '73+']
        X['TenureGroup'] = pd.cut(X['tenure'], bins=tenure_bins, labels=tenure_labels, right=False)
        
        # Indicateur de nouveau client
        X['IsNewCustomer'] = (X['tenure'] <= 6).astype(int)
        
        # Indicateur de client fidèle
        X['IsLoyalCustomer'] = (X['tenure'] >= 36).astype(int)
        
        # Ratio tenure/mois de contrat
        if 'Contract' in X.columns:
            # Conversion du type de contrat en durée en mois
            contract_to_months = {
                'Month-to-month': 1,
                'One year': 12,
                'Two year': 24
            }
            
            X['ContractMonths'] = X['Contract'].map(contract_to_months).fillna(1)
            
            # Calcul du ratio tenure/contrat
            X['TenureContractRatio'] = X['tenure'] / X['ContractMonths']
            
            # Indicateur de renouvellement de contrat
            X['HasRenewedContract'] = (X['TenureContractRatio'] > 1).astype(int)
    
    def _create_charge_features(self, X):
        """
        Crée des fonctionnalités liées aux charges financières
        
        Args:
            X (pd.DataFrame): Données d'entrée
        """
        # Vérifier si les colonnes nécessaires existent
        charge_cols = ['MonthlyCharges', 'TotalCharges']
        missing_cols = [col for col in charge_cols if col not in X.columns]
        
        if missing_cols:
            logger.warning(f"Colonnes manquantes pour les fonctionnalités de charges: {missing_cols}")
            return
        
        # Convertir TotalCharges en numérique si nécessaire
        if X['TotalCharges'].dtype == 'object':
            X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')
        
        # Calcul de la dépense moyenne par mois basée sur la tenure
        if 'tenure' in X.columns:
            X['AvgMonthlySpend'] = X['TotalCharges'] / X['tenure'].replace(0, 1)
            
            # Ratio entre les charges mensuelles actuelles et la moyenne historique
            X['CurrentVsHistoricalChargeRatio'] = X['MonthlyCharges'] / X['AvgMonthlySpend']
            X['CurrentVsHistoricalChargeRatio'].replace([np.inf, -np.inf], 1, inplace=True)
            X['CurrentVsHistoricalChargeRatio'].fillna(1, inplace=True)
        
        # Catégorisation des charges mensuelles
        monthly_charge_bins = [0, 35, 70, 105, np.inf]
        monthly_charge_labels = ['Low', 'Medium', 'High', 'Very High']
        X['MonthlyChargesCategory'] = pd.cut(X['MonthlyCharges'], bins=monthly_charge_bins, labels=monthly_charge_labels)
        
        # Indicateur de client premium basé sur les charges
        X['IsPremiumBySpend'] = (X['MonthlyCharges'] > X['MonthlyCharges'].quantile(0.75)).astype(int)


class FeatureEngineer:
    """Classe principale pour l'ingénierie des fonctionnalités"""
    
    def __init__(self, config):
        """
        Initialise l'ingénieur de fonctionnalités
        
        Args:
            config: Instance de la classe Config
        """
        self.config = config
        self.feature_pipeline = None
    
    def create_pipeline(self):
        """
        Crée un pipeline de transformation des fonctionnalités
        
        Returns:
            Pipeline: Pipeline scikit-learn
        """
        logger.info("Création du pipeline de transformation des fonctionnalités")
        
        self.feature_pipeline = Pipeline([
            ('custom_features', CustomFeatureTransformer())
        ])
        
        return self.feature_pipeline
    
    def transform(self, X):
        """
        Applique la transformation des fonctionnalités
        
        Args:
            X (pd.DataFrame): Données d'entrée
        
        Returns:
            pd.DataFrame: Données transformées
        """
        if self.feature_pipeline is None:
            self.create_pipeline()
        
        logger.info("Application des transformations de fonctionnalités")
        
        return self.feature_pipeline.transform(X)
    
    def fit_transform(self, X, y=None):
        """
        Ajuste et applique la transformation des fonctionnalités
        
        Args:
            X (pd.DataFrame): Données d'entrée
            y: Variable cible (non utilisée)
        
        Returns:
            pd.DataFrame: Données transformées
        """
        if self.feature_pipeline is None:
            self.create_pipeline()
        
        logger.info("Ajustement et application des transformations de fonctionnalités")
        
        return self.feature_pipeline.fit_transform(X, y)
    
    def save_pipeline(self, output_path=None):
        """
        Sauvegarde le pipeline de fonctionnalités
        
        Args:
            output_path (str, optional): Chemin pour sauvegarder le pipeline
        """
        if self.feature_pipeline is None:
            logger.warning("Impossible de sauvegarder: le pipeline n'a pas été créé")
            return
        
        if output_path is None:
            output_path = os.path.join(self.config.get('paths.models_dir'), 'feature_pipeline.pkl')
        
        # Création du répertoire si nécessaire
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        joblib.dump(self.feature_pipeline, output_path)
        
        logger.info(f"Pipeline de fonctionnalités sauvegardé à {output_path}")
    
    def load_pipeline(self, input_path=None):
        """
        Charge un pipeline de fonctionnalités sauvegardé
        
        Args:
            input_path (str, optional): Chemin du pipeline sauvegardé
        """
        if input_path is None:
            input_path = os.path.join(self.config.get('paths.models_dir'), 'feature_pipeline.pkl')
        
        logger.info(f"Chargement du pipeline de fonctionnalités depuis {input_path}")
        
        self.feature_pipeline = joblib.load(input_path)
        
        logger.info("Pipeline de fonctionnalités chargé avec succès")