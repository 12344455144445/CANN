from pathlib import Path
import json, numpy as np

root=Path(__file__).resolve().parent
rng=np.random.default_rng(2026101003)
random=rng.integers(0,0x7f800000,1000000,dtype=np.uint32)
special=np.array([0,1,63,64,126,127,128,0x7fffff,0x800000,0x7f7fffff],dtype=np.uint32)
edges=np.concatenate([np.arange(max(0,(e<<23)-4),(e<<23)+5,dtype=np.uint32) for e in range(1,255)])
mantissa=np.concatenate([np.arange((e<<23)+0x7e0000-4,(e<<23)+0x7e0000+5,dtype=np.uint32) for e in range(1,254)])
bits=np.concatenate([random,special,edges,mantissa])
e=(bits>>23)&255
m=(bits&0x7fffff)|np.where(e==0,0,0x800000).astype(np.uint32)
high=m>=16646144
shift=np.where(e<=7,np.where(e==0,0,e-1),np.where(high,6,7)).astype(np.uint64)
numerator=m.astype(np.uint64)<<shift
q=numerator//127
q+=(numerator-q*127)*2>127
out_e=np.where(e<=7,0,e-np.where(high,6,7)).astype(np.uint64)
carry=(out_e!=0)&(q==0x1000000)
q=np.where(carry,q>>1,q)
out_e+=carry
actual=np.where(out_e==0,q,(out_e<<23)|(q&0x7fffff)).astype(np.uint32)
expected=(bits.view(np.float32)/np.float32(127)).view(np.uint32)
bad=np.flatnonzero(actual!=expected)
assert not bad.size, [(int(bits[i]),int(actual[i]),int(expected[i])) for i in bad[:10]]
result={'positive_finite_bits_tested':len(bits),'status':'PASS','coverage':'Random fp32 incl subnormal, all exponent transitions and mantissa threshold neighbors. CPU proof, not NPU compiler validation.'}
(root/'v70_onepass'/'scale127_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(result)
