from configs import transforms_config
from configs.paths_config import dataset_paths


DATASETS = {
    'ffhq_encode': {
        'transforms': transforms_config.EncodeTransforms,
        'train_source_root': dataset_paths['ffhq'],
        'train_target_root': dataset_paths['ffhq'],
        'test_source_root': dataset_paths['celeba_test'],
        'test_target_root': dataset_paths['celeba_test']
    },
    'ham10k': {
        'transforms': transforms_config.MedTransforms,
        'train_source_root': dataset_paths['ham10k'],
        'train_target_root': dataset_paths['ham10k'],
        'test_source_root': dataset_paths['ham10k'],
        'test_target_root': dataset_paths['ham10k'],
    },
    'rxrx19b': {
        'transforms': transforms_config.MedTransforms,
        'train_source_root': dataset_paths['rxrx19b'],
        'train_target_root': dataset_paths['rxrx19b'],
        'test_source_root': dataset_paths['rxrx19b'],
        'test_target_root': dataset_paths['rxrx19b'],
    }
}