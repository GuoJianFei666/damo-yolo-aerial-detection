"""All fixtures are constructed logic tests, not VisDrone performance results."""
from copy import deepcopy
import hashlib
import math
import pytest
from aerial_quant.data_audit import parse_annotation, image_summary, audit_splits, file_sha256
from aerial_quant.calibration import select_coverage
from aerial_quant.policy import module_names_hash, resolve_policy, conv2d_constructor_kwargs
from aerial_quant.selection import Constraints, quantile, select_measured


def record(image_id, group_id):
    return dict(image_id=image_id, group_id=group_id,
                sha256=hashlib.sha256(image_id.encode()).hexdigest())


def splits():
    training = [record('a', 'flight1'), record('b', 'flight2')]
    return {'train': training, 'calibration': [training[0].copy()],
            'select': [record('c', 'flight3')], 'confirm': [record('d', 'flight4')],
            'test': [record('e', 'flight5')]}


def candidate(name, latency=10., ap=40., small=20., recall=70.):
    return dict(candidate_id=name, data_origin='synthetic_unit_test',
                partition='select', protocol_id='synthetic_protocol',
                hardware_id='NO_HARDWARE_SYNTHETIC', checkpoint_sha256='synthetic-checkpoint',
                engine_sha256=f'synthetic-{name}', evaluation_manifest_sha256='synthetic-manifest',
                timing_scope='decode_excluded_e2e', ap=ap, ap_small=small,
                group_recall={'tiny': recall, 'occluded': recall}, latency_ms=[latency] * 1000)


def test_annotation_keeps_ignored_regions():
    rows = parse_annotation('0,0,20,20,1,1,0,0\n0,0,100,100,0,0,0,0\n5,5,9,9,1,11,0,2\n')
    result = image_summary(rows, 1920, 1080)
    assert result['raw_boxes'] == 3
    assert result['evaluable_boxes'] == 1
    assert result['ignored_or_other_boxes'] == 2
    assert result['tiny_after_resize_boxes'] == 1


@pytest.mark.parametrize('line', [
    '1,2,3', '0,0,0,3,1,1,0,0', '0,0,3,3,1,12,0,0',
    'nan,0,3,3,1,1,0,0', '0,0,3,3,0.5,1,0,0', '0,0,3,3,1,1,0,9'])
def test_annotation_rejects_invalid_rows(line):
    with pytest.raises(ValueError):
        parse_annotation(line)


def test_out_of_image_box_is_reported_not_clipped():
    rows = parse_annotation('90,90,20,20,1,1,0,0')
    assert image_summary(rows, 100, 100)['out_of_image_row_indices'] == [0]
    assert rows[0].width == 20


def test_effective_small_definition_is_resolution_sensitive():
    rows = parse_annotation('0,0,24,24,1,1,0,0')
    assert image_summary(rows, 1280, 1280, 640)['tiny_after_resize_boxes'] == 1
    assert image_summary(rows, 1280, 1280, 1280)['tiny_after_resize_boxes'] == 0


def test_file_hash_reads_bytes(tmp_path):
    p = tmp_path / 'test.bin'; p.write_bytes(b'constructed-fixture')
    assert file_sha256(p) == hashlib.sha256(b'constructed-fixture').hexdigest()


def test_split_calibration_subset_is_allowed():
    assert audit_splits(splits())['status'] == 'passed'


@pytest.mark.parametrize('key', ['image_id', 'sha256', 'group_id'])
def test_split_cross_partition_overlap_is_rejected(key):
    data = splits()
    data['test'][0][key] = data['select'][0][key]
    with pytest.raises(ValueError, match='overlap'):
        audit_splits(data)


def test_calibration_from_test_is_rejected():
    data = splits(); data['calibration'] = deepcopy(data['test'])
    with pytest.raises(ValueError, match='subset'):
        audit_splits(data)


def test_duplicate_image_content_with_new_name_is_rejected():
    data = splits(); row = deepcopy(data['train'][0]); row['image_id'] = 'renamed'
    data['train'].append(row)
    with pytest.raises(ValueError, match='duplicate'):
        audit_splits(data)


def coverage_rows():
    return [dict(image_id=str(i), group_id=f'g{i//2}', split='train', tags=tags)
            for i, tags in enumerate([['tiny'], ['normal'], ['occluded'], ['normal'], ['tiny', 'occluded'], ['normal']])]


def test_coverage_is_deterministic_and_group_bounded():
    rows = coverage_rows()
    selected = select_coverage(rows, 3, {'tiny': 1, 'occluded': 1}, 1, seed=9)
    assert selected == select_coverage(rows, 3, {'tiny': 1, 'occluded': 1}, 1, seed=9)
    selected_rows = [r for r in rows if r['image_id'] in selected]
    assert len({r['group_id'] for r in selected_rows}) == 3
    tags = {t for r in selected_rows for t in r['tags']}
    assert {'tiny', 'occluded'} <= tags


def test_coverage_does_not_admit_test_images():
    rows = coverage_rows(); rows[0]['split'] = 'test'
    with pytest.raises(ValueError, match='train'):
        select_coverage(rows, 3, {'tiny': 1}, 1)


def test_coverage_failure_is_not_silent():
    with pytest.raises(ValueError, match='did not meet coverage'):
        select_coverage(coverage_rows(), 2, {'missing': 1}, 2)


def test_group_cap_failure_is_not_silent():
    with pytest.raises(ValueError, match='group cap'):
        select_coverage(coverage_rows(), 4, {}, 1)


def test_name_policy_is_independent_of_enumeration_order():
    names = ['neck.conv', 'backbone.conv', 'head.conv']
    policy = dict(schema_version=1, module_names_sha256=module_names_hash(names), quantize=['neck.conv'])
    assert resolve_policy(policy, reversed(names)) == ['neck.conv']


@pytest.mark.parametrize('requested', [['missing'], ['neck.conv', 'neck.conv']])
def test_policy_rejects_unknown_or_duplicate_names(requested):
    names = ['neck.conv']
    with pytest.raises(ValueError):
        resolve_policy(dict(schema_version=1, module_names_sha256=module_names_hash(names), quantize=requested), names)


def test_policy_rejects_changed_module_set():
    policy = dict(schema_version=1, module_names_sha256=module_names_hash(['a']), quantize=['a'])
    with pytest.raises(ValueError, match='does not match'):
        resolve_policy(policy, ['a', 'b'])


def test_empty_quantization_policy_is_an_explicit_baseline():
    policy = dict(schema_version=1, module_names_sha256=module_names_hash(['a']), quantize=[])
    assert resolve_policy(policy, ['a']) == []


def test_selector_rejects_synthetic_data_by_default():
    with pytest.raises(ValueError, match='synthetic'):
        select_measured([candidate('fp16')], 'fp16', Constraints())


def test_selector_protects_small_targets():
    rows = [candidate('fp16'), candidate('fast_but_bad', 5, small=17), candidate('feasible', 8, ap=39.5, small=19.5)]
    result = select_measured(rows, 'fp16', Constraints(), allow_synthetic=True)
    assert result['selected']['candidate_id'] == 'feasible'
    assert result['candidates'][0]['rejection_reasons'] == ['small_ap']


def test_selector_protects_worst_group():
    result = select_measured([candidate('fp16'), candidate('bad', 7, recall=65)],
                             'fp16', Constraints(), allow_synthetic=True)
    assert result['status'] == 'no_feasible_candidate'
    assert result['candidates'][0]['rejection_reasons'] == ['group_recall']


def test_selector_reports_failure_instead_of_relaxing_threshold():
    result = select_measured([candidate('fp16'), candidate('slow', 9)],
                             'fp16', Constraints(), allow_synthetic=True)
    assert result['status'] == 'no_feasible_candidate' and result['selected'] is None


@pytest.mark.parametrize('field,value', [('partition', 'test'), ('hardware_id', 'other-device'),
                                        ('ap_small', math.nan)])
def test_selector_rejects_invalid_or_incomparable_measurements(field, value):
    row = candidate('x', 8); row[field] = value
    with pytest.raises(ValueError):
        select_measured([candidate('fp16'), row], 'fp16', Constraints(), allow_synthetic=True)


def test_quantile_matches_linear_interpolation():
    assert quantile([1, 2, 3, 4, 5], 0.95) == pytest.approx(4.8)


def test_latency_requires_enough_samples():
    row = candidate('fp16'); row['latency_ms'] = [10.]
    with pytest.raises(ValueError, match='insufficient'):
        select_measured([row], 'fp16', Constraints(), allow_synthetic=True)


def test_conv_adapter_preserves_groups_dilation_and_numerics():
    import torch
    from torch import nn
    torch.manual_seed(9)
    source = nn.Conv2d(4, 8, 3, groups=2, dilation=2, padding=2, bias=False)
    clone = nn.Conv2d(**conv2d_constructor_kwargs(source))
    clone.load_state_dict(source.state_dict())
    x = torch.randn(1, 4, 12, 12)
    with torch.inference_mode():
        torch.testing.assert_close(source(x), clone(x), rtol=0, atol=0)


def test_upstream_style_constructor_drops_group_information():
    from torch import nn
    source = nn.Conv2d(4, 8, 3, groups=2)
    original_style = nn.Conv2d(source.in_channels, source.out_channels,
                              source.kernel_size, source.stride, source.padding)
    assert original_style.weight.shape != source.weight.shape
    with pytest.raises(RuntimeError):
        original_style.weight.data.copy_(source.weight.detach())
