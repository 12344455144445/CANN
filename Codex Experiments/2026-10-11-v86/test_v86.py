"""CPU arithmetic/layout and static source-scope checks; no NPU execution."""
from pathlib import Path
import unittest
import re
import numpy as np

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'2026-10-11-v83-v85'/'v84'/'kernel.asc'

def body(source,name):
    start=source.index(name+'(');opening=source.index('{',start);end=opening+1;depth=1
    while depth:
        depth+=(source[end]=='{')-(source[end]=='}');end+=1
    return source[opening:end]

class Checks(unittest.TestCase):
    def test_actual_source_fallback_selection(self):
        text=(ROOT/'v86/kernel.asc').read_text(encoding='utf-8')
        def convert(expression):
            return re.sub(r'(0x[0-9a-fA-F]+|\d+)U',r'\1',expression).replace('||',' or ').replace('&&',' and ')
        safe_expr=convert(re.search(r'return (.*);',body(text,'QmqTinyVectorScaleSafe')).group(1))
        def safe(value):
            word=np.asarray(value,dtype=np.float32).view(np.uint32).item()&0x7fffffff
            return bool(eval(safe_expr,{'__builtins__':{}},{'absolute':word}))
        route_expr=convert(re.search(r'const bool vectorPost = (.*);',body(text,'QmqTryTinyDirect')).group(1))
        fixtures=[(8,1.,1.,False),(15,1.,1.,False),(16,1.,1.,True),(32,1.,1.,True),
                  (16,0.,1.,True),(16,-1.,-1.,True),(16,2.**-60,2.**40,True),
                  (16,2.**-61,1.,False),(16,1.,2.**41,False),
                  (16,1.,float('inf'),False),(16,float('nan'),1.,False)]
        for n,s1,s2,want in fixtures:
            route=eval(route_expr,{'__builtins__':{}},{'n':n,'s1':s1,
                'vectorS2Safe':all(safe(s2) for _ in range(n)),
                'QmqTinyVectorScaleSafe':safe})
            self.assertEqual(bool(route),want,(n,s1,s2))

    def test_candidate_scope(self):
        path=ROOT/'v86/kernel.asc'
        self.assertTrue(path.exists(),'v86 candidate missing')
        base=BASE.read_text(encoding='utf-8');text=path.read_text(encoding='utf-8')
        for name in ('QmqDiv','QmqCorrectDiv','QmqQuantizeNearHalf','QmqScaleBy127Exact',
                     'quant_matmul_relu_quant_cube','quant_matmul_relu_quant_vector',
                     'quant_matmul_relu_quant_fused','qmq_tiny_vector_entry','run_kernel'):
            self.assertEqual(body(base,name),body(text,name),name)
        tiny=body(text,'QmqTryTinyDirect')
        self.assertIn('n >= 16U && vectorS2Safe && QmqTinyVectorScaleSafe(s1)',tiny)
        self.assertIn('WholeReduceMax(sums[alignedN], sums, static_cast<int32_t>(n)',tiny)
        self.assertIn('DataCopy(activationLocal[r*activationStride], sums, alignedN)',tiny)
        self.assertIn('volatile float firstScaled=sums.GetValue(j)*s1',tiny)

    def test_scale_guard_literals(self):
        bits=np.array([0,0x80000000,0x21800000,0xa1800000,0x53800000,0xd3800000,
                       1,0x21800000-1,0x53800000+1,0x7f800000,0x7fc00000],np.uint32)
        absolute=bits&np.uint32(0x7fffffff)
        text=(ROOT/'v86/kernel.asc').read_text(encoding='utf-8')
        expression=re.search(r'return (.*);',body(text,'QmqTinyVectorScaleSafe')).group(1)
        expression=re.sub(r'(0x[0-9a-fA-F]+|\d+)U',r'\1',expression)
        expression=expression.replace('||',' or ').replace('&&',' and ')
        safe=[bool(eval(expression,{'__builtins__':{}},{'absolute':int(v)})) for v in absolute]
        np.testing.assert_array_equal(safe,[True,True,True,True,True,True,False,False,False,False,False])

    def test_vector_and_scalar_pipeline(self):
        rng=np.random.default_rng(72420786);tested=0
        for n in range(16,33):
            for k in (1,7,16,31,32,33,64):
                if n*k>1024:continue
                a=rng.integers(-128,128,(7,k),dtype=np.int8)
                b=rng.integers(-128,128,(n,k),dtype=np.int8)
                a[0]=0;a[1]=-128;b[0]=-128
                dots=a.astype(np.int32)@b.astype(np.int32).T
                for exponent in (-60,-30,-7,0,20,40):
                    scale1=np.full(7,np.ldexp(np.float32(1),exponent),np.float32)
                    scale1[2]*=-1;scale1[3]=0
                    scale2=rng.uniform(-1,1,n).astype(np.float32)*np.ldexp(np.float32(1),exponent)
                    # Include explicit minimal normal result and maximal integer-product scales.
                    scale2[0]=np.ldexp(np.float32(1),exponent)
                    expected=np.empty((7,n),np.float32)
                    for r in range(7):
                        for j in range(n):
                            first=np.float32(np.float32(dots[r,j])*scale1[r])
                            value=np.float32(first*scale2[j])
                            expected[r,j]=value if value>0 else np.float32(0)
                    scaled=dots.astype(np.float32)*scale1[:,None]
                    scaled*=scale2
                    actual=np.maximum(scaled,np.float32(0))
                    np.testing.assert_array_equal(actual,expected)
                    maximum=actual.max(axis=1)
                    scale=np.where(maximum>0,maximum/np.float32(127),np.float32(1)).astype(np.float32)
                    ys=np.where(expected.max(axis=1)>0,expected.max(axis=1)/np.float32(127),np.float32(1)).astype(np.float32)
                    np.testing.assert_array_equal(scale.view(np.uint32),ys.view(np.uint32))
                    np.testing.assert_array_equal(np.rint(actual/scale[:,None]),np.rint(expected/ys[:,None]))
                    # Only copy align8(N) floats, clearing any copied dummy lane.
                    aligned=(n+7)//8*8
                    sums=np.full((7,2*aligned),np.nan,np.float32);sums[:,:n]=actual
                    sums[:,n:aligned]=0
                    output=np.zeros((7,32),np.float32);output[:,:aligned]=sums[:,:aligned]
                    np.testing.assert_array_equal(output[:,:n],expected)
                    self.assertTrue(np.all(output[:,n:]==0))
                    tested+=1
        self.assertGreater(tested,500)

if __name__=='__main__':unittest.main(verbosity=2)
