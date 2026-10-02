"""Compatibility for native Cosmos logging with W&B 0.28+."""
def ensure_wandb_generate_id():
    import wandb
    if not hasattr(wandb.util, 'generate_id'):
        from wandb.sdk.lib.runid import generate_id
        wandb.util.generate_id = generate_id
