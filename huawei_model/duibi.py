import sys
import torch

def load_ckpt(path):
    return torch.load(path, map_location='cpu', weights_only=False)

def tensor_info(v):
    if torch.is_tensor(v):
        return {'type': 'tensor', 'shape': tuple(v.shape), 'dtype': str(v.dtype)}
    return {'type': type(v).__name__, 'value': v}

def compare(a_path, b_path):
    a = load_ckpt(a_path)
    b = load_ckpt(b_path)

    if not isinstance(a, dict) or not isinstance(b, dict):
        print('ERROR: checkpoint 不是 dict，无法按 key 对比')
        print(f'A type: {type(a)}, B type: {type(b)}')
        return

    meta_keys = ('epoch', 'step', 'steps')
    a_meta = {k: a[k] for k in meta_keys if k in a}
    b_meta = {k: b[k] for k in meta_keys if k in b}

    print('=' * 60)
    print('META 字段对比')
    print('=' * 60)
    print(f'A: {a_path}')
    print(f'   {a_meta if a_meta else "(无 epoch/step)"}')
    print(f'B: {b_path}')
    print(f'   {b_meta if b_meta else "(无 epoch/step)"}')

    a_keys = set(a.keys())
    b_keys = set(b.keys())
    only_a = sorted(a_keys - b_keys)
    only_b = sorted(b_keys - a_keys)
    common = sorted(a_keys & b_keys)

    print('\n' + '=' * 60)
    print('KEY 统计')
    print('=' * 60)
    print(f'A total keys: {len(a_keys)}')
    print(f'B total keys: {len(b_keys)}')
    print(f'common keys:  {len(common)}')
    print(f'only in A:    {len(only_a)}')
    print(f'only in B:    {len(only_b)}')

    if only_a:
        print('\n--- only in A ---')
        for k in only_a[:30]:
            print(f'  {k}: {tensor_info(a[k])}')
        if len(only_a) > 30:
            print(f'  ... 还有 {len(only_a)-30} 个')

    if only_b:
        print('\n--- only in B ---')
        for k in only_b[:30]:
            print(f'  {k}: {tensor_info(b[k])}')
        if len(only_b) > 30:
            print(f'  ... 还有 {len(only_b)-30} 个')

    shape_diff = []
    dtype_diff = []
    value_diff = []
    for k in common:
        if k in meta_keys:
            if a[k] != b[k]:
                print(f'\nMETA 不同: {k}: A={a[k]}, B={b[k]}')
            continue
        va, vb = a[k], b[k]
        if torch.is_tensor(va) and torch.is_tensor(vb):
            if va.shape != vb.shape:
                shape_diff.append((k, tuple(va.shape), tuple(vb.shape)))
            if va.dtype != vb.dtype:
                dtype_diff.append((k, str(va.dtype), str(vb.dtype)))
            if va.shape == vb.shape and not torch.equal(va, vb):
                value_diff.append(k)
        elif va != vb:
            print(f'\n非 tensor 字段不同: {k}: A={va}, B={vb}')

    print('\n' + '=' * 60)
    print('共同 key 中的结构差异')
    print('=' * 60)
    print(f'shape 不同: {len(shape_diff)}')
    for k, sa, sb in shape_diff[:20]:
        print(f'  {k}: A{sa} vs B{sb}')
    if len(shape_diff) > 20:
        print(f'  ... 还有 {len(shape_diff)-20} 个')

    print(f'\ndtype 不同: {len(dtype_diff)}')
    for k, da, db in dtype_diff[:20]:
        print(f'  {k}: A={da}, B={db}')

    print(f'\nshape 相同但数值不同: {len(value_diff)}')
    for k in value_diff[:20]:
        print(f'  {k}')
    if len(value_diff) > 20:
        print(f'  ... 还有 {len(value_diff)-20} 个')

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('用法: python compare_flow_pt.py <flow_a.pt> <flow_b.pt>')
        sys.exit(1)
    compare(sys.argv[1], sys.argv[2])
