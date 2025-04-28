import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
    classification_report
)
from joblib import dump, load
import json

from utils.logger import get_logger

logger = get_logger()

class ModelEvaluator:
    """Classe pour l'évaluation des modèles de prédiction de churn"""
    
    def __init__(self, config):
        """
        Initialise l'évaluateur de modèles
        
        Args:
            config: Instance de la classe Config
        """
        self.config = config
        self.output_dir = config.get('paths.output_dir')
        
        # Création du répertoire de sortie si nécessaire
        os.makedirs(self.output_dir, exist_ok=True)
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name, save=True):
        """
        Trace et sauvegarde la matrice de confusion
        
        Args:
            y_true: Valeurs réelles
            y_pred: Prédictions
            model_name (str): Nom du modèle
            save (bool): Si True, sauvegarde le graphique
        
        Returns:
            matplotlib.figure.Figure: Figure matplotlib
        """
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalisation de la matrice
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Création de la heatmap
        sns.heatmap(
            cm_norm, 
            annot=cm,
            fmt='d', 
            cmap='Blues', 
            cbar=False,
            xticklabels=['Non Churn', 'Churn'],
            yticklabels=['Non Churn', 'Churn']
        )
        
        plt.title(f'Matrice de Confusion - {model_name}')
        plt.ylabel('Classe Réelle')
        plt.xlabel('Classe Prédite')
        
        if save:
            output_path = os.path.join(self.output_dir, f'confusion_matrix_{model_name}.png')
            plt.tight_layout()
            plt.savefig(output_path)
            logger.info(f"Matrice de confusion sauvegardée à {output_path}")
        
        return plt.gcf()
    
    def plot_roc_curve(self, y_true, y_prob, model_names=None, save=True):
        """
        Trace et sauvegarde la courbe ROC
        
        Args:
            y_true: Valeurs réelles
            y_prob: Liste de probabilités prédites pour chaque modèle
            model_names (list): Liste des noms de modèles
            save (bool): Si True, sauvegarde le graphique
        
        Returns:
            matplotlib.figure.Figure: Figure matplotlib
        """
        plt.figure(figsize=(10, 8))
        
        # Si un seul modèle est fourni
        if not isinstance(y_prob, list):
            y_prob = [y_prob]
            
        if model_names is None:
            model_names = [f"Modèle {i+1}" for i in range(len(y_prob))]
        elif not isinstance(model_names, list):
            model_names = [model_names]
        
        # Tracer la courbe ROC pour chaque modèle
        for i, (probs, name) in enumerate(zip(y_prob, model_names)):
            fpr, tpr, _ = roc_curve(y_true, probs)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(
                fpr, tpr, 
                lw=2, 
                label=f'{name} (AUC = {roc_auc:.3f})'
            )
        
        # Ligne de référence (modèle aléatoire)
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taux de Faux Positifs')
        plt.ylabel('Taux de Vrais Positifs')
        plt.title('Courbe ROC')
        plt.legend(loc="lower right")
        
        if save:
            output_path = os.path.join(self.output_dir, 'roc_curve_comparison.png')
            plt.tight_layout()
            plt.savefig(output_path)
            logger.info(f"Courbe ROC sauvegardée à {output_path}")
        
        return plt.gcf()
    
    def plot_precision_recall_curve(self, y_true, y_prob, model_names=None, save=True):
        """
        Trace et sauvegarde la courbe précision-rappel
        
        Args:
            y_true: Valeurs réelles
            y_prob: Liste de probabilités prédites pour chaque modèle
            model_names (list): Liste des noms de modèles
            save (bool): Si True, sauvegarde le graphique
        
        Returns:
            matplotlib.figure.Figure: Figure matplotlib
        """
        plt.figure(figsize=(10, 8))
        
        # Si un seul modèle est fourni
        if not isinstance(y_prob, list):
            y_prob = [y_prob]
            
        if model_names is None:
            model_names = [f"Modèle {i+1}" for i in range(len(y_prob))]
        elif not isinstance(model_names, list):
            model_names = [model_names]
        
        # Tracer la courbe précision-rappel pour chaque modèle
        for i, (probs, name) in enumerate(zip(y_prob, model_names)):
            precision, recall, _ = precision_recall_curve(y_true, probs)
            pr_auc = auc(recall, precision)
            
            plt.plot(
                recall, precision, 
                lw=2, 
                label=f'{name} (AUC = {pr_auc:.3f})'
            )
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Rappel')
        plt.ylabel('Précision')
        plt.title('Courbe Précision-Rappel')
        plt.legend(loc="lower left")
        
        if save:
            output_path = os.path.join(self.output_dir, 'precision_recall_curve_comparison.png')
            plt.tight_layout()
            plt.savefig(output_path)
            logger.info(f"Courbe précision-rappel sauvegardée à {output_path}")
        
        return plt.gcf()
    
    def plot_feature_importance(self, model, feature_names, model_name, top_n=20, save=True):
        """
        Trace et sauvegarde l'importance des fonctionnalités
        
        Args:
            model: Modèle entraîné avec attribut feature_importances_
            feature_names (list): Noms des fonctionnalités
            model_name (str): Nom du modèle
            top_n (int): Nombre de fonctionnalités les plus importantes à afficher
            save (bool): Si True, sauvegarde le graphique
        
        Returns:
            matplotlib.figure.Figure: Figure matplotlib
        """
        # Vérifier si le modèle a un attribut feature_importances_
        if not hasattr(model, 'feature_importances_'):
            logger.warning(f"Le modèle {model_name} n'a pas d'attribut feature_importances_")
            return None
        
        # Obtenir les importances et les trier
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Limiter aux top_n fonctionnalités
        indices = indices[:top_n]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]
        
        # Tracer l'importance des fonctionnalités
        plt.figure(figsize=(12, 8))
        plt.title(f'Importance des Fonctionnalités - {model_name}')
        plt.barh(range(len(top_importances)), top_importances, align='center')
        plt.yticks(range(len(top_importances)), top_features)
        plt.xlabel('Importance')
        plt.ylabel('Fonctionnalité')
        plt.tight_layout()
        
        if save:
            output_path = os.path.join(self.output_dir, f'feature_importance_{model_name}.png')
            plt.savefig(output_path)
            logger.info(f"Importance des fonctionnalités sauvegardée à {output_path}")
        
        return plt.gcf()
    
    def plot_probability_distribution(self, y_true, y_prob, model_name, save=True):
        """
        Trace et sauvegarde la distribution des probabilités prédites
        
        Args:
            y_true: Valeurs réelles
            y_prob: Probabilités prédites
            model_name (str): Nom du modèle
            save (bool): Si True, sauvegarde le graphique
        
        Returns:
            matplotlib.figure.Figure: Figure matplotlib
        """
        plt.figure(figsize=(10, 6))
        
        # Distribution pour les exemples positifs (churn)
        plt.hist(
            y_prob[y_true == 1], 
            alpha=0.5, 
            bins=20, 
            range=(0, 1),
            label='Churn'
        )
        
        # Distribution pour les exemples négatifs (non churn)
        plt.hist(
            y_prob[y_true == 0], 
            alpha=0.5, 
            bins=20, 
            range=(0, 1),
            label='Non Churn'
        )
        
        plt.xlabel('Probabilité Prédite de Churn')
        plt.ylabel('Nombre d\'exemples')
        plt.title(f'Distribution des Probabilités Prédites - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save:
            output_path = os.path.join(self.output_dir, f'probability_distribution_{model_name}.png')
            plt.tight_layout()
            plt.savefig(output_path)
            logger.info(f"Distribution des probabilités sauvegardée à {output_path}")
        
        return plt.gcf()
    
    def generate_classification_report(self, y_true, y_pred, model_name, output_format='txt'):
        """
        Génère et sauvegarde un rapport de classification
        
        Args:
            y_true: Valeurs réelles
            y_pred: Prédictions
            model_name (str): Nom du modèle
            output_format (str): Format de sortie ('txt' ou 'json')
        
        Returns:
            dict or str: Rapport de classification
        """
        report = classification_report(y_true, y_pred, output_dict=(output_format == 'json'))
        
        # Sauvegarde du rapport
        if output_format == 'json':
            output_path = os.path.join(self.output_dir, f'classification_report_{model_name}.json')
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=4)
        else:  # txt
            output_path = os.path.join(self.output_dir, f'classification_report_{model_name}.txt')
            with open(output_path, 'w') as f:
                f.write(report)
        
        logger.info(f"Rapport de classification sauvegardé à {output_path}")
        
        return report
    
    def evaluate_model(self, model, X_test, y_test, feature_names=None, model_name=None):
        """
        Évalue un modèle de manière complète
        
        Args:
            model: Modèle à évaluer
            X_test: Caractéristiques de test
            y_test: Variable cible de test
            feature_names (list, optional): Noms des fonctionnalités
            model_name (str, optional): Nom du modèle
        
        Returns:
            dict: Résultats de l'évaluation
        """
        if model_name is None:
            model_name = type(model).__name__
        
        logger.info(f"Évaluation complète du modèle: {model_name}")
        
        # Récupération des prédictions
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            # Pour les modèles comme Keras qui renvoient directement des probabilités
            try:
                y_prob = model.predict(X_test).ravel()
            except:
                y_prob = None
                logger.warning(f"Impossible d'obtenir des probabilités pour le modèle {model_name}")
        
        y_pred = model.predict(X_test)
        if len(y_pred.shape) > 1:
            y_pred = y_pred.ravel()
        
        if y_prob is not None:
            y_pred_binary = (y_prob > 0.5).astype(int)
        else:
            y_pred_binary = y_pred
        
        # Générer les visualisations
        self.plot_confusion_matrix(y_test, y_pred_binary, model_name)
        
        if y_prob is not None:
            self.plot_roc_curve(y_test, y_prob, model_name)
            self.plot_precision_recall_curve(y_test, y_prob, model_name)
            self.plot_probability_distribution(y_test, y_prob, model_name)
        
        # Importance des fonctionnalités si disponible
        if hasattr(model, 'feature_importances_') and feature_names is not None:
            self.plot_feature_importance(model, feature_names, model_name)
        
        # Génération du rapport de classification
        report = self.generate_classification_report(y_test, y_pred_binary, model_name, output_format='json')
        
        return {
            'model_name': model_name,
            'classification_report': report,
            'y_pred': y_pred_binary,
            'y_prob': y_prob
        }
    
    def compare_models(self, models_dict, X_test, y_test):
        """
        Compare plusieurs modèles
        
        Args:
            models_dict (dict): Dictionnaire de modèles {nom: modèle}
            X_test: Caractéristiques de test
            y_test: Variable cible de test
        
        Returns:
            pd.DataFrame: Tableau de comparaison des modèles
        """
        logger.info(f"Comparaison de {len(models_dict)} modèles")
        
        model_names = []
        y_probs = []
        
        results = {}
        
        # Évaluer chaque modèle et collecter les résultats
        for name, model in models_dict.items():
            logger.info(f"Évaluation du modèle: {name}")
            
            # Prédictions
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                try:
                    y_prob = model.predict(X_test).ravel()
                except:
                    y_prob = None
                    logger.warning(f"Impossible d'obtenir des probabilités pour le modèle {name}")
            
            if y_prob is not None:
                y_pred = (y_prob > 0.5).astype(int)
                model_names.append(name)
                y_probs.append(y_prob)
            else:
                y_pred = model.predict(X_test)
                if len(y_pred.shape) > 1:
                    y_pred = y_pred.ravel()
            
            # Récupération des métriques de base
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
            }
            
            if y_prob is not None:
                metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
            
            results[name] = metrics
        
        # Tracer les courbes de comparaison si des probabilités sont disponibles
        if y_probs:
            self.plot_roc_curve(y_test, y_probs, model_names)
            self.plot_precision_recall_curve(y_test, y_probs, model_names)
        
        # Création du DataFrame de comparaison
        comparison_df = pd.DataFrame.from_dict(results, orient='index')
        
        # Sauvegarde du tableau de comparaison
        output_path = os.path.join(self.output_dir, 'model_comparison.csv')
        comparison_df.to_csv(output_path)
        logger.info(f"Tableau de comparaison des modèles sauvegardé à {output_path}")
        
        return comparison_df