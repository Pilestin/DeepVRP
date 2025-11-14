



""" Buradaki fonksiyon(lar) verilen matris dosyalarını okuyup numpy array formatına dönüştürür. """

import pandas as pd
import numpy as np
import ast


def read_distance_matrix(file_path: str):
    """
    Distance matrix dosyasını okur ve numpy array döndürür.
    
    Returns:
        numpy.ndarray: Distance matrix (NxN)
    """
    df = pd.read_excel(file_path, index_col=0)
    matrix = df.values.astype(np.float32)
    print(f"✓ Distance matrix loaded: {matrix.shape}")
    return matrix


def read_energy_matrix(file_path: str):
    """
    Energy matrix dosyasını okur ve numpy array döndürür.
    
    Returns:
        numpy.ndarray: Energy matrix (NxN)
    """
    df = pd.read_excel(file_path, index_col=0)
    matrix = df.values.astype(np.float32)
    print(f"✓ Energy matrix loaded: {matrix.shape}")
    return matrix


def read_location_matrix(file_path: str):
    """
    Location matrix dosyasını okur ve numpy array döndürür.
    Location matrix içindeki koordinat listelerini parse eder.
    
    Returns:
        dict: Her node için koordinat bilgileri
    """
    df = pd.read_excel(file_path, index_col=0)
    
    # Location matrix'in formatını kontrol et
    # Her hücre string olarak koordinat listesi içerebilir
    location_data = {}
    
    for idx in df.index:
        for col in df.columns:
            cell_value = df.loc[idx, col]
            if pd.notna(cell_value) and cell_value != '[]':
                try:
                    # String'i liste olarak parse et
                    coords = ast.literal_eval(str(cell_value))
                    if isinstance(coords, list) and len(coords) > 0:
                        location_data[(idx, col)] = np.array(coords, dtype=np.float32)
                except:
                    pass
    
    print(f"✓ Location matrix loaded: {len(location_data)} location paths")
    return location_data