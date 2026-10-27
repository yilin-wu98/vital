"""Differentiable tactile-language scores; these are not calibrated probabilities."""
import torch
import torch.nn.functional as F


def pool_tokens(tokens, aggregate="mean_patch"):
    """Pool AnyTouch output (CLS, five sensor tokens, then patch tokens)."""
    if tokens.ndim < 3 or tokens.shape[-2] <= 6:
        raise ValueError("Expected (..., 6 + patch_count, embedding_dim) tokens")
    if aggregate == "cls":
        pooled = tokens[..., 0, :]
    elif aggregate == "mean_patch":
        pooled = tokens[..., 6:, :].mean(dim=-2)
    elif aggregate == "max_patch":
        pooled = tokens[..., 6:, :].max(dim=-2).values
    elif aggregate == "min_patch":
        pooled = tokens[..., 6:, :].min(dim=-2).values
    else:
        raise ValueError(f"Unknown token aggregation: {aggregate}")
    return F.normalize(pooled, dim=-1)


def score_patch_embeddings(patches, text_embeddings, negative_text_embeddings=None):
    """Score WM patch predictions against text; preserve gradients through patches.

    ``patches`` is (..., patches, dim), with no CLS or sensor tokens.
    ``text_embeddings`` is (prompts, dim) or (dim,).
    The result is (..., prompts). With negative text, return positive minus
    negative cosine similarity, as used by the two-phase liquid experiments.
    """
    if patches.ndim < 3 or patches.shape[-2] == 0:
        raise ValueError("Expected a nonempty batch of patch embeddings")
    texts = torch.atleast_2d(text_embeddings)
    if texts.ndim != 2 or texts.shape[-1] != patches.shape[-1]:
        raise ValueError("Text and tactile embedding dimensions must match")
    pooled = F.normalize(patches.mean(dim=-2), dim=-1)
    scores = pooled @ F.normalize(texts, dim=-1).T
    if negative_text_embeddings is not None:
        negative = torch.atleast_2d(negative_text_embeddings)
        if negative.shape != texts.shape:
            raise ValueError("Positive and negative text embeddings must have the same shape")
        scores = scores - pooled @ F.normalize(negative, dim=-1).T
    return scores
