"""Checks on actual tensors returned by the native blender."""


def blend_layer_audit(result, indices, layer_id, settings):
    """Verify the layer0 ablation actually enters the native sparse branch."""
    query_tokens, cache_tokens = result[0].shape[0], result[1].shape[0]
    positions = None if indices is None else indices.tolist()
    if settings.get('check_layer') == 0:
        expected = max(int(cache_tokens * settings['ratio']), 1)
        if positions is None or len(positions) != expected or query_tokens != expected:
            raise RuntimeError(f'Layer0 ablation did not apply sparse KV replacement at layer {layer_id}')
    return {'layer': layer_id, 'positions': positions,
            'method': settings['method'], 'requested_ratio': settings['ratio'],
            'check_layer': settings['check_layer'], 'query_tokens': query_tokens,
            'cache_tokens': cache_tokens, 'updated_tokens': 0 if positions is None else len(positions)}
