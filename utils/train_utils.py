import os
from pathlib import Path


class ModelPathReorganizer:
    def __init__(self, save_dir: Path, metric_name: str = 'accuracy', reverse: bool = True):
        """
        Args:
            save_dir (Path): Directory where models are saved.
            metric_name (str): The metric to sort by (e.g., 'accuracy').
            reverse (bool): If True, higher metric is better.
        """
        self.save_dir = save_dir.resolve().absolute()
        self.metric_name = metric_name
        self.reverse = reverse
        self.models = {}  # abs path -> metric_value

    def update(self, model_path: Path, metric_value: float):
        """Add or update a model's metric."""
        model_path = model_path.resolve().absolute()
        self.models[model_path] = metric_value

    def sort_and_rename(self):
        """Sort models by metric and rename them according to their rank."""
        sorted_models = sorted(self.models.items(), key=lambda item: item[1], reverse=self.reverse)

        # First pass: rename to temporary files to avoid overwrites
        temp_paths = {}
        for i, (path, metric_value) in enumerate(sorted_models):
            temp_path = path.with_name(f'_temp_model_{i}.pt').resolve().absolute()
            os.rename(path, temp_path)
            temp_paths[temp_path] = metric_value

        # Second pass: rename from temp to final destination
        self.models.clear()
        for i, (temp_path, metric_value) in enumerate(temp_paths.items()):
            final_path = (self.save_dir / f'model_{i}.pt').resolve().absolute()
            os.rename(temp_path, final_path)
            self.models[final_path] = metric_value
