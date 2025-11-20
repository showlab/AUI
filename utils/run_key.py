from hashlib import sha1


def _slug_initial(initial_dir: str) -> str:
    if not initial_dir:
        return "default"
    return str(initial_dir).replace("/", "_").replace(" ", "-")


def build_run_key(revision_type: str, commenter: str, initial_dir: str) -> str:
    """Deterministic namespace key for a Stage 3 configuration.

    Format: rev-<revision_type>__commenter-<commenter>__initial-<slug>
    """
    return f"rev-{revision_type}__commenter-{commenter}__initial-{_slug_initial(initial_dir)}"


def short_run_key(run_key: str) -> str:
    return sha1(run_key.encode("utf-8")).hexdigest()[:8]
