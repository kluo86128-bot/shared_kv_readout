from types import SimpleNamespace

import pytest
import torch

from runtime_fixes.positions import install, CURRENT, POSITION_ATTR, source_positions


def rotate(x, positions):
    half = x.shape[-1]//2
    angles = positions.float()[:, None] * (1./(10000.**(torch.arange(half).float()/half)))[None]
    cos, sin = angles.cos()[:, None], angles.sin()[:, None]
    first, second = x[..., :half], x[..., half:]
    return torch.cat((first*cos-second*sin, second*cos+first*sin), -1)


def classes(audit):
    class Fused:
        def __call__(self, old, new, key):
            return rotate(rotate(key, -old), new)

    class Connector:
        def __init__(self):
            self.fused = Fused()

        def batched_from_gpu(self, objects, starts, ends, **kwargs):
            # Emulate stale/zero cached_positions in the serving integration.
            for layer in objects:
                for obj in layer:
                    obj.metadata.cached_positions = torch.zeros(obj.tensor.shape[1], dtype=torch.long)
            yield 'stored'

        def batched_to_gpu(self, starts, ends, **kwargs):
            for _ in range(2):
                objects = yield 'ready'
                old = torch.cat([o.metadata.cached_positions for o in objects])
                new = torch.arange(starts[0], ends[-1])
                key = torch.cat([o.tensor[0] for o in objects]).reshape(len(new), 1, -1)
                yield self.fused(old, new, key)

    install(Connector, Fused, audit)
    return Connector, Fused


def test_native_send_path_corrects_219_plus_2j_to_reader_j():
    rows = []
    Connector, _ = classes(lambda layer, seen, used, new, corrected:
                           rows.append((layer, seen.clone(), used.clone(), corrected)))
    connector = Connector()
    torch.manual_seed(17)
    raw = torch.randn(56, 1, 8)
    source = torch.arange(219, 275)
    objects = [[SimpleNamespace(tensor=torch.stack((rotate(raw, source).flatten(1), torch.zeros(56, 8))),
                                metadata=SimpleNamespace())] for _ in range(2)]
    assert list(connector.batched_from_gpu(objects, [219], [275])) == ['stored']
    consumer = connector.batched_to_gpu([0], [56])
    assert next(consumer) == 'ready'
    for layer in range(2):
        result = consumer.send(objects[layer])
        assert torch.allclose(result, rotate(raw, torch.arange(56)), atol=1e-6)
        assert CURRENT.get() is None
        if layer == 0:
            assert next(consumer) == 'ready'
    consumer.close()
    assert [r[0] for r in rows] == [0, 1]
    assert all(r[3] and torch.equal(r[2], source) for r in rows)
    assert all(torch.equal(r[1], torch.zeros(56, dtype=torch.long)) for r in rows)


def test_no_transfer_keeps_lmcache_startup_self_test_unchanged():
    Connector, Fused = classes(None)
    key = torch.randn(4, 1, 8)
    old = torch.arange(4)
    new = old + 20
    assert torch.equal(Fused()(old, new, key), rotate(rotate(key, -old), new))


def test_missing_origin_positions_is_rejected_instead_of_guessed():
    Connector, _ = classes(None)
    consumer = Connector().batched_to_gpu([0], [4])
    next(consumer)
    obj = SimpleNamespace(tensor=torch.zeros(2, 4, 8), metadata=SimpleNamespace())
    with pytest.raises(RuntimeError, match='verified source positions'):
        consumer.send([obj])
    assert CURRENT.get() is None
    consumer.close()


def test_saved_source_positions_can_be_restored_with_cache_objects(tmp_path):
    Connector, _ = classes(None)
    obj = SimpleNamespace(tensor=torch.zeros(2, 4, 8), metadata=SimpleNamespace())
    list(Connector().batched_from_gpu([[obj]], [219], [223]))
    path = tmp_path/'positions.pt'
    torch.save(source_positions(obj), path)
    restored = SimpleNamespace(tensor=obj.tensor.clone())
    setattr(restored, POSITION_ATTR, torch.load(path, weights_only=True))
    assert torch.equal(source_positions(restored), torch.arange(219, 223))


def test_writer_trailing_empty_segment_is_a_valid_native_store():
    # Actual spans reproduced with the frozen tokenizer for the first parity
    # writer. The chat generation prompt ends in delimiter token 271.
    starts = [0, 40, 219, 276, 427, 429]
    ends = [39, 218, 275, 426, 428, 429]
    Connector, _ = classes(None)
    objects = [[SimpleNamespace(tensor=torch.zeros(2, end-start, 8), metadata=SimpleNamespace())
                for start, end in zip(starts, ends)] for _ in range(2)]
    assert list(Connector().batched_from_gpu(objects, starts, ends)) == ['stored']
    for layer in objects:
        assert source_positions(layer[-1]).shape == (0,)
        assert source_positions(layer[-1]).dtype == torch.long
        assert torch.equal(source_positions(layer[2]), torch.arange(219, 275))


@pytest.mark.parametrize('start,end,shape', [
    (-1, 3, (2, 4, 8)), (4, 3, (2, 0, 8)), (0, 4, (2, 3, 8)),
])
def test_invalid_native_store_spans_still_fail_with_details(start, end, shape):
    Connector, _ = classes(None)
    obj = SimpleNamespace(tensor=torch.zeros(shape), metadata=SimpleNamespace())
    with pytest.raises(RuntimeError, match=r'layer=0, chunk=0, start='):
        list(Connector().batched_from_gpu([[obj]], [start], [end]))
