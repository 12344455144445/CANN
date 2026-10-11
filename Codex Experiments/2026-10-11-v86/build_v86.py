from pathlib import Path
import hashlib,json,difflib

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'2026-10-11-v83-v85'/'v84/kernel.asc'
EXPECTED='b17cf7e9e7da2dca60ffbefd35851c769dadaa1e53c764d301dc608265b07ebc'

def replace_once(text,old,new):
    assert text.count(old)==1,old[:90]
    result=text.replace(old,new)
    assert result.replace(new,old)==text
    return result

def build():
    assert hashlib.sha256(BASE.read_bytes()).hexdigest()==EXPECTED
    base=BASE.read_text(encoding='utf-8');text=base
    marker='// Integer dot products fit int32 and fp32 exactly because K<=128.'
    helper='''// Bounded finite scales keep each nonzero scaled integer normal and finite.
// INT8 dot magnitude <= 2^21; scales are zero or in [2^-60, 2^40].
// The two separate FP32 multiplies therefore stay in [2^-120, 2^101].
__aicore__ inline bool QmqTinyVectorScaleSafe(float value)
{
    union { float value; uint32_t bits; } word;
    word.value = value;
    const uint32_t absolute = word.bits & 0x7fffffffU;
    return absolute == 0U || (absolute >= 0x21800000U && absolute <= 0x53800000U);
}

'''
    text=replace_once(text,marker,helper+marker)
    text=replace_once(text,
        '    for(uint32_t j=0;j<n;++j){s2Values[j]=s2Local.GetValue(j);}',
        '''    bool vectorS2Safe = true;
    for(uint32_t j=0;j<n;++j){
        s2Values[j]=s2Local.GetValue(j);
        vectorS2Safe = vectorS2Safe && QmqTinyVectorScaleSafe(s2Values[j]);
    }''')
    old='''        float maximum=0.0f;
        for(uint32_t j=0;j<n;++j){
            volatile float firstScaled=sums.GetValue(j)*s1;
            const float dequantized=firstScaled*s2Values[j];
            union { float value; uint32_t bits; } check;
            check.value=dequantized;
            if((check.bits&0x7f800000U)==0x7f800000U){return false;}
            activations[j]=dequantized>0.0f?dequantized:0.0f;
            if(activations[j]>maximum){maximum=activations[j];}
        }
        const float rowScale=maximum>0.0f?QmqScaleBy127Exact(maximum):1.0f;
        if(!(rowScale>0.0f)){return false;}
        ysLocal.SetValue(r,rowScale);
        for(uint32_t j=0;j<n;++j){
            activationLocal.SetValue(r*activationStride+j,activations[j]);
        }'''
    new='''        float maximum=0.0f;
        const bool vectorPost = n >= 16U && vectorS2Safe && QmqTinyVectorScaleSafe(s1);
        if (vectorPost) {
            Muls(sums, sums, s1, n);
            PipeBarrier<PIPE_V>();
            Mul(sums, sums, s2Local, n);
            PipeBarrier<PIPE_V>();
            Maxs(sums, sums, 0.0f, n);
            PipeBarrier<PIPE_V>();
            WholeReduceMax(sums[alignedN], sums, static_cast<int32_t>(n),
                1, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
            PipeBarrier<PIPE_ALL>();
            maximum = sums.GetValue(alignedN);
            for (uint32_t j = n; j < alignedN; ++j) {
                sums.SetValue(j, 0.0f);
            }
            PipeBarrier<PIPE_ALL>();
            DataCopy(activationLocal[r*activationStride], sums, alignedN);
            PipeBarrier<PIPE_V>();
        } else {
            for(uint32_t j=0;j<n;++j){
                volatile float firstScaled=sums.GetValue(j)*s1;
                const float dequantized=firstScaled*s2Values[j];
                union { float value; uint32_t bits; } check;
                check.value=dequantized;
                if((check.bits&0x7f800000U)==0x7f800000U){return false;}
                activations[j]=dequantized>0.0f?dequantized:0.0f;
                if(activations[j]>maximum){maximum=activations[j];}
            }
        }
        const float rowScale=maximum>0.0f?QmqScaleBy127Exact(maximum):1.0f;
        if(!(rowScale>0.0f)){return false;}
        ysLocal.SetValue(r,rowScale);
        if (!vectorPost) {
            for(uint32_t j=0;j<n;++j){
                activationLocal.SetValue(r*activationStride+j,activations[j]);
            }
        }'''
    text=replace_once(text,old,new)
    text='// V86: tiny vector postprocessing candidate from V84; device validation pending.\n'+text
    folder=ROOT/'v86';folder.mkdir(exist_ok=True)
    path=folder/'kernel.asc';path.write_bytes(text.replace('\n','\r\n').encode('utf-8'))
    (ROOT/'v86/vs_v84.diff').write_text(''.join(difflib.unified_diff(base.splitlines(True),text.splitlines(True),fromfile='v84',tofile='v86')),encoding='utf-8')
    manifest={'base_version':'v84','base_sha256':EXPECTED,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
              'bytes':path.stat().st_size,'npu_status':'PENDING','gate':'tiny; N>=16; all scales zero or magnitude in [2^-60,2^40]',
              'fallback':'Original V84 scalar postprocessing for N<16 or out-of-range scales.'}
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':build()
