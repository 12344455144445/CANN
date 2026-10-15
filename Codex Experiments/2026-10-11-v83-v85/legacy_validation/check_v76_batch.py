from pathlib import Path
import json,numpy as np
root=Path(__file__).resolve().parent
rng=np.random.default_rng(2026101076)
count=0;values=0
for rows in range(1,97):
    for n in range(1,33):
        scales=np.exp2(rng.uniform(-70,50,rows)).astype(np.float32)
        q=rng.uniform(0,127,(rows,n)).astype(np.float32)
        # Seed exact half neighborhoods across independent row scales.
        q[:,0]=rng.integers(0,127,rows).astype(np.float32)+np.float32(.5)
        a=q*scales[:,None]
        a[::7,:]=0
        expected=np.clip(np.rint(a/scales[:,None]),0,127).astype(np.int8)
        acts=np.zeros((rows,32),np.float32);acts[:,:n]=a
        divisors=np.repeat(scales[:,None],32,axis=1)
        approx=acts/divisors
        residue=np.abs(approx-(np.floor(approx)+np.float32(.5)))
        selected=residue<np.float32(.00002)
        flat_selected=np.flatnonzero(selected)
        for lane in flat_selected:
            assert lane//32<rows and lane%32<n
            assert divisors.ravel()[lane]==scales[lane//32]
        # Bit-mask iteration groups cover each active candidate once.
        compare=(rows*32+63)//64*64
        seen=[]
        for group in range(compare//64):
            lanes=flat_selected[(flat_selected//64)==group]
            mask=sum(1<<int(l%64) for l in lanes)
            while mask:
                bit=(mask&-mask).bit_length()-1
                lane=group*64+bit
                if lane<rows*32 and lane%32<n:seen.append(lane)
                mask &= mask-1
        assert seen==flat_selected.tolist()
        ints=np.clip(np.rint(approx),0,127).astype(np.int32)
        half=ints.astype(np.float16)
        out=half.astype(np.int8)
        assert np.array_equal(out[:,:n],expected)
        assert np.all(out[:,n:]==0)
        packed=np.concatenate([out[r,:n] for r in range(rows)])
        assert np.array_equal(packed.reshape(rows,n),expected)
        # Overwriting expired activation storage must not overlap half scratch storage.
        activation_bytes=acts.copy().view(np.uint8).ravel()
        activation_bytes.view(np.int32)[:]=ints.ravel()
        scratch_bytes=np.zeros(compare*4,np.uint8)
        scratch_bytes.view(np.float16)[:rows*32]=activation_bytes.view(np.int32).astype(np.float16)
        assert np.array_equal(scratch_bytes.view(np.float16)[:rows*32].astype(np.int8).reshape(rows,32),out)
        count+=1;values+=rows*n
report={'status':'PASS','row_N_batch_configurations':count,'true_output_values':values,
        'coverage':'Every rows 1..96 and N 1..32; per-row scales, half boundaries, zero rows, mask words, odd-row compare tail, int/half storage and GM packing',
        'limitations':'CPU layout/correctness model; original near-half functions unchanged. Hardware compiler, DMA behavior, sync and timings still require platform regression.'}
(root/'v76_tiny_batch_quant'/'batch_checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(report)
