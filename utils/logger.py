import logging
import os
import sys
from datetime import datetime

def setup_logger(log_file=None):
    """
    Configure le logger pour le projet
    
    Args:
        log_file (str, optional): Chemin du fichier de log. Si None, les logs sont écrits sur la sortie standard.
    
    Returns:
        logging.Logger: Logger configuré
    """
    logger = logging.getLogger('churn_prediction')
    logger.setLevel(logging.INFO)
    
    # Format pour les logs
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Handler pour la console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    
    # Handler pour fichier si spécifié
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(log_dir=None):
    """
    Récupère ou crée un logger avec un nom de fichier basé sur la date actuelle
    
    Args:
        log_dir (str, optional): Répertoire pour les fichiers de log. Si None, les logs sont écrits sur la sortie standard.
    
    Returns:
        logging.Logger: Logger configuré
    """
    if log_dir:
        # Création d'un nom de fichier basé sur la date
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'churn_prediction_{current_time}.log')
    else:
        log_file = None
    
    return setup_logger(log_file)