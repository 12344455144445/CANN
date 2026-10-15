from pathlib import Path
import json, numpy as np
root=Path(__file__).resolve().parent
rng=np.random.default_rng(2026101072)

def round_exact(activation,scale):
    a=np.asarray(activation,dtype=np.float32)
    s=np.broadcast_to(np.asarray(scale,dtype=np.float32),a.shape).copy()
    aw=a.view(np.uint32);sw=s.view(np.uint32)
    ae=(aw>>23)&255;se=(sw>>23)&255
    am=(aw&0x7fffff)|np.where(ae==0,0,0x800000).astype(np.uint32)
    sm=(sw&0x7fffff)|np.where(se==0,0,0x800000).astype(np.uint32)
    shift=np.where(ae==0,1,ae).astype(np.int32)-np.where(se==0,1,se).astype(np.int32)
    clipped=np.clip(shift,-1,8)
    numerator=am.astype(np.uint64)<<np.maximum(clipped,0).astype(np.uint64)
    denominator=sm.astype(np.uint64)<<np.maximum(-clipped,0).astype(np.uint64)
    lower=np.minimum(numerator//denominator,127).astype(np.uint32)
    half=lower.astype(np.float32)+np.float32(.5)
    hw=half.view(np.uint32)
    hm=(hw&0x7fffff)|0x800000
    threshold=hm.astype(np.uint64)*2+np.where((lower&1)==0,1,-1).astype(np.int64).astype(np.uint64)
    threshold_exp=((hw>>23)&255).astype(np.int32)-151
    sh=np.maximum(clipped-threshold_exp,0).astype(np.uint64)
    lhs=am.astype(np.uint64)<<sh
    rhs=sm.astype(np.uint64)*threshold
    add=np.where((lower&1)==0,lhs>rhs,lhs>=rhs)
    output=np.minimum(lower+add,127).astype(np.int32)
    output=np.where(shift<=-2,0,output)
    output=np.where(shift>=9,127,output)
    return np.where(a>0,output,0).astype(np.int8)

maxbits=rng.integers(1,0x7f800000,400000,dtype=np.uint32)
actbits=(rng.random(maxbits.size)*maxbits).astype(np.uint32)
maxima=maxbits.view(np.float32)
scales=maxima/np.float32(127)
valid=(scales>0)&np.isfinite(scales)
a=actbits[valid].view(np.float32);s=scales[valid]
expected=np.clip(np.rint(a/s),0,127).astype(np.int8)
actual=round_exact(a,s)
bad=np.flatnonzero(actual!=expected)
assert bad.size==0,[(a[i],s[i],actual[i],expected[i]) for i in bad[:5]]

scale_bits=rng.integers(1,0x7b000000,25000,dtype=np.uint32)
ss=scale_bits.view(np.float32)
lowers=rng.integers(0,127,len(ss),dtype=np.int32)
centers=(lowers.astype(np.float32)+np.float32(.5))*ss
around=np.concatenate([np.nextafter(centers,-np.inf,dtype=np.float32),centers,np.nextafter(centers,np.inf,dtype=np.float32)])
ts=np.tile(ss,3)
expected2=np.clip(np.rint(around/ts),0,127).astype(np.int8)
actual2=round_exact(around,ts)
bad=np.flatnonzero(actual2!=expected2)
assert not bad.size,[(around[i],ts[i],actual2[i],expected2[i]) for i in bad[:10]]
count=0
for m in [1,3,7,16,96]:
    for n in [1,7,8,16,32]:
        for k in [1,16,32,64,128]:
            if n*k>1024:continue
            x1=rng.integers(-128,128,(m,k),dtype=np.int8)
            x2=rng.integers(-128,128,(n,k),dtype=np.int8)
            if m>1:x1[-1]=0
            s1=rng.uniform(-2,2,m).astype(np.float32)
            s2=rng.uniform(-2,2,n).astype(np.float32)
            c=x1.astype(np.int32)@x2.astype(np.int32).T
            a=np.maximum((c.astype(np.float32)*s1[:,None])*s2[None,:],np.float32(0))
            mx=a.max(axis=1)
            ys=np.where(mx>0,mx/np.float32(127),np.float32(1))
            y=np.clip(np.rint(a/ys[:,None]),0,127).astype(np.int8)
            assert np.array_equal(round_exact(a,ys[:,None]),y)
            count+=1
report={'random_quantization_values':int(valid.sum()),'near_half_values':len(around),'matrix_shapes':count,'status':'PASS','limitation':'CPU integer-threshold arithmetic validation; NPU compiler, launch, and scalar multiplication order require platform validation.'}
(root/'v72_tiny_direct'/'tiny_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(report)
