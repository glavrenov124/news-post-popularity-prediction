import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "cointegrated/rubert-tiny2"


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class RuBertEmbedder:
    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None):
        self.model_name = model_name
        self.device = device or get_device()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return (token_embeddings * input_mask_expanded).sum(
            1
        ) / input_mask_expanded.sum(1).clamp(min=1e-9)

    @torch.no_grad()
    def encode_texts(
        self,
        texts: list[str],
        batch_size: int = 64,
        max_length: int = 128,
    ) -> np.ndarray:
        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i : i + batch_size]

            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            outputs = self.model(**encoded)

            emb = self.mean_pooling(outputs, encoded["attention_mask"])
            emb = F.normalize(emb, p=2, dim=1)
            all_embeddings.append(emb.cpu().numpy())

        return np.vstack(all_embeddings)


def build_rubert_embeddings(
    df: pd.DataFrame,
    text_col: str = "text",
    batch_size: int = 64,
    max_length: int = 128,
    model_name: str = MODEL_NAME,
) -> pd.DataFrame:
    embedder = RuBertEmbedder(model_name=model_name)

    embeddings = embedder.encode_texts(
        texts=df[text_col].tolist(),
        batch_size=batch_size,
        max_length=max_length,
    )

    emb_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
    emb_df = pd.DataFrame(embeddings, columns=emb_cols)
    emb_df["row_id"] = df["row_id"].values

    return emb_df
