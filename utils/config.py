import os
import yaml
from dotenv import load_dotenv

# Chargement des variables d'environnement
load_dotenv()

class Config:
    """Classe pour gérer la configuration du projet"""
    
    def __init__(self, config_file=None):
        """
        Initialise la configuration du projet
        
        Args:
            config_file (str, optional): Chemin vers le fichier de configuration YAML
        """
        self.config = {}
        
        # Chemins par défaut
        self.config['paths'] = {
            'data_dir': os.path.join(os.getcwd(), 'data'),
            'models_dir': os.path.join(os.getcwd(), 'models'),
            'logs_dir': os.path.join(os.getcwd(), 'logs'),
            'output_dir': os.path.join(os.getcwd(), 'output')
        }
        
        # Paramètres des modèles par défaut
        self.config['model_params'] = {
            'logistic_regression': {
                'C': 1.0,
                'max_iter': 100,
                'solver': 'liblinear',
                'random_state': 42
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'random_state': 42
            },
            'neural_network': {
                'batch_size': 32,
                'epochs': 50,
                'validation_split': 0.2,
                'patience': 10
            }
        }
        
        # Paramètres de prétraitement par défaut
        self.config['preprocessing'] = {
            'test_size': 0.2,
            'random_state': 42,
            'scaling': 'standard',  # 'standard', 'minmax', ou None
            'handle_imbalance': 'class_weight'  # 'none', 'class_weight', 'smote'
        }
        
        # Chargement de la configuration depuis un fichier s'il est fourni
        if config_file and os.path.exists(config_file):
            self._load_from_file(config_file)
        
        # Créer les répertoires nécessaires
        self._create_directories()
    
    def _load_from_file(self, config_file):
        """
        Charge la configuration depuis un fichier YAML
        
        Args:
            config_file (str): Chemin vers le fichier de configuration YAML
        """
        with open(config_file, 'r') as f:
            file_config = yaml.safe_load(f)
            # Mise à jour récursive de la configuration
            self._update_dict(self.config, file_config)
    
    def _update_dict(self, d, u):
        """
        Met à jour récursivement un dictionnaire
        
        Args:
            d (dict): Dictionnaire à mettre à jour
            u (dict): Dictionnaire avec les nouvelles valeurs
        """
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = self._update_dict(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    
    def _create_directories(self):
        """Crée les répertoires nécessaires pour le projet"""
        for _, path in self.config['paths'].items():
            os.makedirs(path, exist_ok=True)
    
    def get(self, key, default=None):
        """
        Récupère une valeur de configuration
        
        Args:
            key (str): Clé de configuration (peut être imbriquée avec des points)
            default: Valeur par défaut si la clé n'existe pas
        
        Returns:
            La valeur de configuration ou la valeur par défaut
        """
        keys = key.split('.')
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def save(self, config_file):
        """
        Sauvegarde la configuration dans un fichier YAML
        
        Args:
            config_file (str): Chemin du fichier de sortie
        """
        with open(config_file, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)