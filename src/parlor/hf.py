"""HuggingFace downloads with an offline fallback to the local cache."""


def download(repo: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download  # deferred — slow import
    try:
        return hf_hub_download(repo, filename)
    except Exception:  # offline — use the local cache
        return hf_hub_download(repo, filename, local_files_only=True)
