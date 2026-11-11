"""Exercise selection, actual KV replacement and gradients, not only CLI parsing."""
import copy
import random

import pytest
import torch

from kv_cache_readout.c1_selection import random_indices, check_indices, validate
from kv_cache_readout.config import load_config, validate_config
from kv_cache_readout.launch import parser, effective_config
from kv_cache_readout.native_blend import native_cacheblend, relocate_keys
from kv_cache_readout.c1_batch import native_cacheblend_many
from kv_cache_readout.legacy_recomputation import random_order
from kv_cache_readout.optim import parameters_for
from kv_cache_readout.training import batch_gradients
from test_c1_batch import pair, inputs
from test_native_autograd import native_backend
from test_joint_readout import tiny_joint


def test_random_budget_is_stable_nested_and_independent_of_global_rng():
    first=random_indices(128,.3,17,'example',torch.device('cpu'))
    assert first.tolist()==sorted(random_order(128,17,'example')[:38])
    random.seed(313);torch.manual_seed(521)
    assert torch.equal(first,random_indices(128,.3,17,'example',torch.device('cpu')))
    assert set(first.tolist())<set(random_indices(128,.5,17,'example',torch.device('cpu')).tolist())
    assert not torch.equal(first,random_indices(128,.3,18,'example',torch.device('cpu')))
    assert not torch.equal(first,random_indices(128,.3,17,'other',torch.device('cpu')))
    for ratio,count in [(0,0),(.001,1),(1,128)]:
        indices=random_indices(128,ratio,17,'example',torch.device('cpu'))
        assert len(indices)==count
        check_indices(indices,128,ratio,torch.device('cpu'))
    for indices in (torch.tensor([3,2]),torch.tensor([2,2]),torch.tensor([2,128]),torch.tensor([2.,3.])):
        with pytest.raises(ValueError):check_indices(indices,128,2/128,torch.device('cpu'))
    for seed in (-1,True,1.5):
        with pytest.raises(ValueError):validate(dict(method='random',seed=seed))


@pytest.mark.parametrize('method',['cacheblend','random','epic'])
@pytest.mark.parametrize('ratio',[0,.1,.3,1])
def test_cli_can_set_each_method_and_ratio(method,ratio):
    args=parser('train').parse_args(['--config','configs/train_hr_joint_main_c128.json',
        '--selection-method',method,'--recompute-ratio',str(ratio),
        '--gpus','0,1,3','--sample-batch-size','4','--global-batch-size','8','--then-offline'])
    cfg,ids=effective_config(args)
    assert ids==['0','1','3']
    assert cfg['c1_selection']==dict(method=method,seed=17)
    assert cfg['native_contract']['recompute_ratio']==ratio
    assert cfg['joint_loss']['fields']['birth_date']['reduction']=='token_mean'
    assert 'readout' not in cfg  # v5 is joint-only, no fieldwise tasks.
    assert cfg['training']['sample_batch_size']==4 and cfg['training']['batch_records']==8


def test_default_none_and_legacy_default_are_both_explicitly_preserved():
    p=parser('train')
    cfg,_=effective_config(p.parse_args(['--config','configs/train_hr_joint_main_c128.json']))
    assert cfg['c1_selection']['method']=='none' and cfg['native_contract']['recompute_ratio']==0
    old,_=effective_config(p.parse_args(['--config','configs/train_hr_joint_c128.json']))
    assert 'c1_selection' not in old and old['native_contract']['recompute_ratio']==.3
    cfg,_=effective_config(p.parse_args(['--config','configs/train_hr_joint_c128.json','--selection-method','none']))
    assert cfg['native_contract']['recompute_ratio']==0
    with pytest.raises(ValueError,match='none means zero'):
        effective_config(p.parse_args(['--config','configs/train_hr_joint_main_c128.json','--recompute-ratio','.3']))
    cfg['native_contract']['recompute_ratio']=.3
    with pytest.raises(ValueError,match='none selection'):validate_config(cfg)


def test_random_packed_matches_serial_kv_gradients_and_isolates_writers(pair):
    b,records,initial=pair
    prompt=initial.clone().requires_grad_()
    public,writers,starts=inputs(b,records,prompt)
    selected=torch.stack([random_indices(56,.3,17,r['record_id'],b.device) for r in records])
    options=dict(check_layer=b.cfg['native_contract']['check_layer'],ratio=.3)
    serial=[native_cacheblend(b.model,public,w,writer_position_start=s,selection_indices=i,**options)
            for w,s,i in zip(writers,starts,selected)]
    packed=native_cacheblend_many(b.model,public,writers,writer_position_starts=starts,
                                selection_indices=selected,**options)
    for one,many,indices in zip(serial,packed,selected):
        assert torch.equal(one.selected,indices) and torch.equal(many.selected,indices)
        assert many.audit['algorithm']=='native_random_sparse'
        for la,lb in zip(one.layers,many.layers):
            for x,y in zip(la,lb):torch.testing.assert_close(x,y,atol=2e-5,rtol=2e-5)
    def objective(results):return sum(t.square().sum() for r in results for pair in r.layers for t in pair)
    gs,=torch.autograd.grad(objective(serial),prompt,retain_graph=True)
    gp,=torch.autograd.grad(objective(packed),prompt)
    torch.testing.assert_close(gs,gp,atol=5e-5,rtol=5e-4)
    detached=[tuple(tuple(t.detach().requires_grad_() for t in layer) for layer in w) for w in writers]
    isolated=native_cacheblend_many(b.model,public,detached,writer_position_starts=starts,
                                   selection_indices=selected,**options)
    other=[t for layer in detached[1] for t in layer]
    grads=torch.autograd.grad(objective(isolated[:1]),other,allow_unused=True)
    assert all(g is None or g.count_nonzero()==0 for g in grads)


@pytest.mark.parametrize('method,ratio',[('none',0),('epic',0),('epic',.3),('epic',1)])
def test_none_and_epic_have_exact_relocated_writer_and_clean_prefix_values(method,ratio):
    b,records,prompt=tiny_joint()
    b.cfg['c1_selection']=dict(method=method,seed=17)
    b.cfg['native_contract']['recompute_ratio']=ratio
    with torch.no_grad():
        writers,suffixes=b.source_many(records,prompt)
        contexts=b.blend_many(records,prompt)
        _,clean=b._model_forward(b.embedding(b.tensor(b.public_ids)),position_start=0)
    count=int(128*ratio)
    for context,writer,suffix in zip(contexts,writers,suffixes):
        start=suffix.start-128-len(b.delimiter_ids)-16
        relocated=relocate_keys(b.model.model,writer,start,0)
        assert context.audit['selection_method']==method
        assert context.audit['selected_indices']==list(range(count))
        for actual,old,fresh in zip(context.layers,relocated,clean):
            for a,o,f in zip(actual,old,fresh):
                torch.testing.assert_close(a[:,:,count:],o[:,:,count:],atol=0,rtol=0)
                if count:torch.testing.assert_close(a[:,:,:count],f[:,:,:count],atol=0,rtol=0)


@pytest.mark.parametrize('method,ratio',[('cacheblend',.3),('cacheblend',1),('random',.3),('random',1),('random',0),
                                      ('epic',.3),('epic',1),('none',0)])
def test_joint_training_all_methods_have_finite_partitioned_gradients(method,ratio):
    b,records,initial=tiny_joint()
    b.cfg['scoped_cache']['preservation_weight']=.1
    b.cfg['c1_selection']=dict(method=method,seed=17)
    b.cfg['native_contract']['recompute_ratio']=ratio
    validate_config(b.cfg)
    grad,loss,active=batch_gradients(b,parameters_for(initial,b.cfg),records,2,0)
    assert torch.isfinite(grad).all() and torch.isfinite(torch.tensor(loss))
    assert all(active)
    if ratio==1:
        # Existing S objective also uses this ratio: full S replacement removes its U signal.
        assert grad[48:].count_nonzero()==0
        assert grad[:48].count_nonzero()==0  # Full clean replacement removes U's QA signal.
    else:
        assert grad[48:].abs().sum()>0
        assert grad[:48].abs().sum()>0


@pytest.mark.parametrize('method',['random','epic','none'])
def test_unsupported_real_engine_export_cannot_silently_run_cacheblend(method):
    from e2e_cache.deployment import validate_native_contract
    cfg=load_config('configs/train_hr_joint_main_c128.json')
    cfg['c1_selection']['method']=method
    with pytest.raises(ValueError,match='Only CacheBlend'):
        validate_native_contract({},cfg)
