"""Dataset loader for KolektorSDD images and task parameter generation."""

import os
import random
from typing import Dict, List, Optional

from PIL import Image

class KolektorSDDLoader:
    """Load KolektorSDD images and derive TaskDAG subtask parameters."""

    def __init__(
        self,
        dataset_path: str,
        seed: Optional[int] = None,
        allow_dummy_data: bool = True,
    ):
        """Initialize the loader.

        Args:
            dataset_path: Root path to the KolektorSDD dataset.
            seed: Optional seed for the loader-private task sampler. The loader
                never draws from the global `random` stream so that the task
                workload is identical across algorithms regardless of how much
                randomness an agent consumes.
            allow_dummy_data: Permit synthetic tasks when the dataset is
                missing or contains no input images.
        """
        self.dataset_path: str = dataset_path
        self.allow_dummy_data = bool(allow_dummy_data)
        self.image_paths: List[str] = self._index_dataset()
        self.random_generator: random.Random = random.Random(seed)

    def reseed(self, seed: int) -> None:
        """Reset the loader-private task sampler.

        Args:
            seed: Seed applied to the loader-private generator.
        """
        self.random_generator = random.Random(seed)

    def _index_dataset(self) -> List[str]:
        """Scan the dataset directory and collect image file paths.

        Returns:
            List of absolute file paths for valid image files.
        """
        valid_extensions = (".jpg", ".png", ".bmp")
        image_paths: List[str] = []
        
        if not os.path.exists(self.dataset_path):
            if not self.allow_dummy_data:
                raise FileNotFoundError(
                    f"Dataset path '{self.dataset_path}' was not found. "
                    "Pass allow_dummy_data=True only for an intentional "
                    "synthetic smoke run."
                )
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

        if not image_paths and not self.allow_dummy_data:
            raise FileNotFoundError(
                f"Dataset path '{self.dataset_path}' contains no input images. "
                "Pass allow_dummy_data=True only for an intentional "
                "synthetic smoke run."
            )
        return image_paths

    def get_random_task_parameters(self) -> Dict[str, Dict[str, float]]:
        """Generate subtask parameters from a random image.

        Returns:
            Mapping of subtask keys to data_size, result_size, and cpu_cycles.
        """
        if not self.image_paths:
            # Fallback to dummy data if the dataset isn't downloaded yet
            file_size_bits = self.random_generator.uniform(1e6, 5e6)  # 1 to 5 Megabits
            pixels = self.random_generator.randint(500000, 2000000)
        else:
            # Load actual image properties
            img_path = self.random_generator.choice(self.image_paths)
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
        
        # Parallel stages 2 and 3 use complementary profiles while preserving
        # their previous combined CPU demand (250 cycles per pixel).
        task_params = {
            "subtask_1": {  # Image Extraction
                "data_size": file_size_bits,
                "result_size": raw_bits,
                "cpu_cycles": pixels * 50,
            },
            "subtask_2": {  # Image Denoising
                "data_size": raw_bits,
                "result_size": raw_bits,
                "cpu_cycles": pixels * 50,
            },
            "subtask_3": {  # Standardization
                "data_size": raw_bits,
                "result_size": raw_bits * 0.2,
                "cpu_cycles": pixels * 250,
            },
            "subtask_4": {  # Feature Extraction
                "data_size": raw_bits * 0.3,
                "result_size": file_size_bits * 0.1,
                "cpu_cycles": pixels * 450,
            },
            "subtask_5": {  # Detection and Recognition
                "data_size": file_size_bits * 0.05,
                "result_size": 256,
                "cpu_cycles": pixels * 250,
            },
        }
        
        return task_params

    def get_dataset_statistics(self) -> Dict[str, object]:
        """Return dataset statistics used in experiments.

        Returns:
            Mapping with dataset counts and alignment with the paper.
        """
        pixel_counts: List[int] = []
        for image_path in self.image_paths:
            with Image.open(image_path) as image:
                pixel_counts.append(image.width * image.height)

        total = len(pixel_counts)
        return {
            "dataset_path": os.path.abspath(self.dataset_path),
            "mode": "real" if total else "dummy",
            "total_images": total,
            "paper_expected_total_images": 399,
            "is_paper_count_aligned": total == 399,
            "min_pixels": min(pixel_counts) if pixel_counts else None,
            "mean_pixels": (
                float(sum(pixel_counts) / total) if pixel_counts else None
            ),
            "max_pixels": max(pixel_counts) if pixel_counts else None,
        }
