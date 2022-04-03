import hydra
from omegaconf import DictConfig, OmegaConf
import argparse
import os
import pprint
from typing import Dict
import pickle

import numpy as np
import pytorch_lightning as pl
import torch
from src.data.processing_utils.dataset_utils import prepare_dataset, prepare_dataloader
from src.data.features import features
from rdkit import Chem

CONFIG_PATH = "../configs"
CONFIG_NAME = "config"
SPLIT_DATASET_PICKLE_FILE_PATH = '/home/ks877/Project/SSLProteins/src/bin/split_dataset.pkl'

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg : DictConfig) -> None:
    # run the following the first time to download the dataset 
    # train_dataset, validation_dataset, test_dataset = prepare_dataset(cfg)
    # split_dataset = {'train': train_dataset, 'valid': validation_dataset, 'test': test_dataset}
    # pickle.dump(split_dataset, open(SPLIT_DATASET_PICKLE_FILE_PATH, 'wb'))

    split_dataset = pickle.load(open(SPLIT_DATASET_PICKLE_FILE_PATH, 'rb'))
    # target, mol, label, structure: target_coord, target_types, ligand_coords
    train_dataset, valid_dataset, test_dataset = split_dataset['train'], split_dataset['valid'], split_dataset['test']

    feat_type = features.FeatureType("mol_graph")
    mol_feat = features.create_feature(feature_type=feat_type, config=cfg)
    feat_type = features.FeatureType("masif")
    protein_feat = features.create_feature(
        feature_type=feat_type, config=cfg
    )

    input_feats = {"protein": [protein_feat], "ligand": [mol_feat]}

    # Get data loaders
    train_loader, valid_loader, test_loader = prepare_dataloader(
        config=cfg,
        feature_inputs=input_feats,
        train_dataset=train_dataset if 'train' in cfg.params.task else None,
        valid_dataset=valid_dataset if 'valid' in cfg.params.task else None,
        test_dataset=test_dataset if 'test' in cfg.params.task else None,
    )

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def test(cfg : DictConfig) -> None:
    split_dataset = pickle.load(open(SPLIT_DATASET_PICKLE_FILE_PATH, 'rb'))
    # target, mol, label, structure: target_coord, target_types, ligand_coords
    train_dataset, valid_dataset, test_dataset = split_dataset['train'], split_dataset['valid'], split_dataset['test']
    test = None
    print(test)
    pdb_id = '1bjv'
    for data in train_dataset:
        if data['target'] == pdb_id:
            test = data
            break
    if not test:
        for data in valid_dataset:
            if data['target'] == pdb_id:
                test = data
                break

    if not test:
        for data in test_dataset:
            if data['target'] == pdb_id:
                test = data
                break
    if test:
        for key, value in test.items():
            print("============")
            print(key)
            print(value)
        
        print(test['structure']['target_coords'].shape)
        print(test['structure']['target_types'].shape)
        print(test['structure']['ligand_coords'].shape)
        print(Chem.rdmolfiles.MolToSmiles(test['mol']))

        
if __name__ == "__main__":
    main()