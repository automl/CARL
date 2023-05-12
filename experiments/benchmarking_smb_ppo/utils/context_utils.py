from .context_sampling import ContextSampler


def get_contexts(cfg, is_eval=False):
    if is_eval:
        if cfg.eval_contexts:
            return cfg.eval_contexts
    else:
        if cfg.contexts:
            return cfg.contexts
    return ContextSampler(**cfg.context_sampler).sample_contexts()
