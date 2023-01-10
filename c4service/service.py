from typing import Optional, Tuple, List, Union

import torch

from c4service.model import C4Model
from c4service.utils import source_process


class ModelService:
    def __init__(self, load_model_path: Optional[str] = None) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.classifier = C4Model()
        if load_model_path:
            self.classifier.load_state_file(load_model_path)

        self.classifier.to(self.device)
        self.classifier.eval()
        self.cosine_similarity = torch.nn.functional.cosine_similarity

    def _preprocess_code(
        self,
        code: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ids_temp, mask_temp = source_process(code)
        source_ids = torch.tensor([ids_temp], dtype=torch.long).to(self.device)
        source_mask = torch.tensor([mask_temp], dtype=torch.long).to(self.device)
        return source_ids, source_mask

    def predict(
        self,
        source_code: str,
        target_code: str,
        return_vector: bool = False,
    ) -> Union[float, Tuple[float, List[float], List[float]]]:
        source_ids, source_mask = self._preprocess_code(source_code)
        target_ids, target_mask = self._preprocess_code(target_code)
        with torch.no_grad():
            sen_vec1, sen_vec2 = self.classifier(
                source_ids,
                source_mask,
                target_ids,
                target_mask,
            )
            
            cos = self.cosine_similarity(sen_vec1, sen_vec2)
            value = cos.cpu().numpy()[0]
            
            if return_vector:
                vec1 = map(float, list(sen_vec1.cpu().numpy()[0]))
                vec2 = map(float, list(sen_vec2.cpu().numpy()[0]))
                return float(value), list(vec1), list(vec2)

            return float(value)

    def predict_vector(self, source_code: str) -> List[float]:
        source_ids, source_mask = self._preprocess_code(source_code)
        with torch.no_grad():
            vec = self.classifier(
                source_ids,
                source_mask,
                return_single=True,
            )
            value = map(float, list(vec.cpu().numpy()[0]))

            return list(value)
