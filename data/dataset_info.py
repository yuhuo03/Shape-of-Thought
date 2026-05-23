
from .interleave_datasets import UnifiedEditIterableDataset
from .t2i_dataset import T2IIterableDataset
from .vlm_dataset import SftJSONLIterableDataset
from .interleave_datasets.sot_dataset import ShapeOfThoughtDataset


DATASET_REGISTRY = {
    't2i_pretrain': T2IIterableDataset,
    'vlm_sft': SftJSONLIterableDataset,
    'unified_edit': UnifiedEditIterableDataset,
    'sot': ShapeOfThoughtDataset,
}


DATASET_INFO = {
    'sot': {
        'sot_dataset': {
            'data_dir': '/path/to/SoT',  # Base directory containing SoT data
            'jsonl_path': '/path/to/SoT/SoT.jsonl',  # Path to SoT.jsonl file
            'image_prefix_dir': '/path/to/SoT',  # Base path for relative image paths
            'num_total_samples': 25929,  # Based on your SoT.jsonl file
        },
    },
}