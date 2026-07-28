"""Dataset loader for KolektorSDD images and task parameter generation."""

import os
import random
from typing import Dict, List

from PIL import Image

class KolektorSDDLoader:
    """Load KolektorSDD images and derive TaskDAG subtask parameters."""

    def __init__(self, dataset_path: str):
        """Initialize the loader.

        Args:
            dataset_path: Root path to the KolektorSDD dataset.
        """
        self.dataset_path: str = dataset_path
        self.image_paths: List[str] = self._index_dataset()

    def _index_dataset(self) -> List[str]:
        """Scan the dataset directory and collect image file paths.

        Returns:
            List of absolute file paths for valid image files.
        """
        valid_extensions = (".jpg", ".png", ".bmp")
        image_paths: List[str] = []
        
        if not os.path.exists(self.dataset_path):
            print(f"Warning: Dataset path '{self.dataset_path}' not found.")
            print("Running in dummy mode for testing.")
            return []

        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                file_stem, file_extension = os.path.splitext(file.lower())
                is_input_image = (
                    file_extension in valid_extensions
                    and not file_stem.endswith("_label")
                )
                if is_input_image:
                    image_paths.append(os.path.join(root, file))
                    
        return image_paths

    def get_random_task_parameters(self) -> Dict[str, Dict[str, float]]:
        """Generate subtask parameters from a random image.

        Returns:
            Mapping of subtask keys to data_size, result_size, and cpu_cycles.
        """
        if not self.image_paths:
            # Fallback to dummy data if the dataset isn't downloaded yet
            file_size_bits = random.uniform(1e6, 5e6)  # 1 to 5 Megabits
            pixels = random.randint(500000, 2000000)
        else:
            # Load actual image properties
            img_path = random.choice(self.image_paths)
            file_size_bytes = os.path.getsize(img_path)
            file_size_bits = file_size_bytes * 8
            
            with Image.open(img_path) as img:
                width, height = img.size
                pixels = width * height

        # The paper outlines specific subtasks for image recognition:
        # 1. Image Extraction, 2. Denoising, 3. Standardization, 
        # 4. Feature Extraction, 5. Detection & Recognition
        
        # We model the data sizes (D), result sizes (R), and CPU cycles (C) 
        # proportionally based on the real image size and pixel count.
        raw_bits = pixels * 8
        
        task_params = {
            "subtask_1": {  # Image Extraction
                "data_size": file_size_bits,
                "result_size": raw_bits,
                "cpu_cycles": pixels * 25,
            },
            "subtask_2": {  # Image Denoising
                "data_size": raw_bits,
                "result_size": raw_bits,
                "cpu_cycles": pixels * 200,
            },
            "subtask_3": {  # Standardization
                "data_size": raw_bits,
                "result_size": raw_bits * 0.3,
                "cpu_cycles": pixels * 50,
            },
            "subtask_4": {  # Feature Extraction
                "data_size": raw_bits * 0.6,
                "result_size": file_size_bits * 0.1,
                "cpu_cycles": pixels * 500,
            },
            "subtask_5": {  # Detection and Recognition
                "data_size": file_size_bits * 0.03,
                "result_size": 256,
                "cpu_cycles": pixels * 300,
            },
        }
        
        return task_params

    def get_dataset_statistics(self) -> Dict[str, int]:
        """Return dataset statistics used in experiments.

        Returns:
            Mapping with dataset counts and alignment with the paper.
        """
        total = len(self.image_paths)
        return {
            "total_images": total,
            "paper_expected_total_images": 399,
            "is_paper_count_aligned": total == 399,
        }
