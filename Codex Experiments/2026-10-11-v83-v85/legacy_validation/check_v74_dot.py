from pathlib import Path
import json,numpy as np
root=Path(__file__).resolve().parent
rng=np.random.default_rng(2026101074)
count=0;max_product_count=0
for n in range(1,33):
    for k in range(1,129):
        if n*k>1024:continue
        stride=(k+31)//32*32
        first=min(k,64)
        # Check active memory addresses for both vector repeats and reduction repeats.
        addresses=[]
        for start,length in [(0,first)]+([(64,k-64)] if k>64 else []):
            for column in range(n):
                addressed=column*stride+start+np.arange(length)
                assert addressed.min()>=column*stride and addressed.max()<(column+1)*stride
                addresses.extend(addressed.tolist())
        assert len(set(addresses))==n*k
        assert max(addresses)<n*stride
        max_product_count=max(max_product_count,n*stride)
        for extreme in [False,True]:
            a=rng.integers(-128,128,(3,k),dtype=np.int8)
            b=rng.integers(-128,128,(n,k),dtype=np.int8)
            if extreme:a[0,:]=-128;b[0,:]=-128;a[-1,:]=0
            padded_a=np.zeros((3,stride),np.int8);padded_b=np.zeros((n,stride),np.int8)
            padded_a[:,:k]=a;padded_b[:,:k]=b
            af=padded_a.astype(np.float16).astype(np.float32)
            bf=padded_b.astype(np.float16).astype(np.float32)
            # SIMD fp32 products and separate per-column reductions.
            products=af[:,None,:]*bf[None,:,:]
            dots=products[:,:,:first].sum(axis=-1,dtype=np.float32)
            if k>64:dots+=products[:,:,64:k].sum(axis=-1,dtype=np.float32)
            expected=a.astype(np.int32)@b.astype(np.int32).T
            assert np.array_equal(dots,expected.astype(np.float32)),(n,k)
            count+=1
report={'status':'PASS','dot_layout_and_numeric_cases':count,'max_padded_B_or_product_elements':max_product_count,
        'coverage':'Every gated N,K pair; random and extremal -128 inputs, K crossing 32/64/96/128, zero rows',
        'proof':'Each int8 and half input exact in fp32; integer products and partial sums bounded by 2097152 and exact in any reduction tree',
        'limitation':'CPU layout and arithmetic proof. ASC overloads, DMA synchronization, compiler resources and hardware performance require platform validation.'}
(root/'v74_tiny_simd_dot'/'dot_checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(report)
