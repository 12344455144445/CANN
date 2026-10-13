"""Interpret actual candidate gate expressions; check batch layout and UB bounds.

Offline source audit only. It does not compile or execute Ascend C instructions.
"""
from pathlib import Path
import json,re,hashlib,sys

class Expression:
    priority={'||':1,'&&':2,'==':3,'!=':3,'<':4,'<=':4,'>':4,'>=':4,'+':5,'-':5,'*':6,'/':6,'%':6}
    def __init__(self,text):
        self.tokens=re.findall(r'\d+U?|[A-Za-z_]\w*|\|\||&&|==|!=|<=|>=|[()+*/%<>,?:!-]',text)
        self.i=0
        self.tree=self.parse(0)
        assert self.i==len(self.tokens),(text,self.tokens[self.i:])
    def take(self):
        token=self.tokens[self.i];self.i+=1;return token
    def peek(self):return self.tokens[self.i] if self.i<len(self.tokens) else None
    def parse(self,minimum):
        token=self.take()
        if token=='(':
            node=self.parse(0);assert self.take()==')'
        elif token in ('!','-'):
            node=('unary',token,self.parse(7))
        elif token[0].isdigit():node=('literal',int(token.rstrip('U')))
        elif self.peek()=='(':
            self.take();args=[self.parse(0)]
            while self.peek()==',':self.take();args.append(self.parse(0))
            assert self.take()==')';node=('call',token,args)
        else:node=('name',token)
        while self.peek() in self.priority and self.priority[self.peek()]>=minimum:
            op=self.take();node=('binary',op,node,self.parse(self.priority[op]+1))
        if minimum==0 and self.peek()=='?':
            self.take();yes=self.parse(0);assert self.take()==':'
            node=('ternary',node,yes,self.parse(0))
        return node
    def value(self,env):
        def visit(t):
            kind=t[0]
            if kind=='literal':return t[1]
            if kind=='name':return {'true':True,'false':False,**env}[t[1]]
            if kind=='unary':return not visit(t[2]) if t[1]=='!' else -visit(t[2])
            if kind=='ternary':return visit(t[2] if visit(t[1]) else t[3])
            if kind=='call':
                args=[visit(x) for x in t[2]]
                if t[1]=='QmqMinValue':return min(args)
                if t[1]=='QmqAlignUp':return (args[0]+args[1]-1)//args[1]*args[1]
                raise ValueError(t[1])
            op=t[1];a=visit(t[2]);b=visit(t[3])
            return {'+':lambda:a+b,'-':lambda:a-b,'*':lambda:a*b,'/':lambda:a//b,'%':lambda:a%b,
                    '&&':lambda:bool(a and b),'||':lambda:bool(a or b),'==':lambda:a==b,'!=':lambda:a!=b,
                    '<':lambda:a<b,'<=':lambda:a<=b,'>':lambda:a>b,'>=':lambda:a>=b}[op]()
        return visit(self.tree)

def body(text,name):
    start=text.index(name+'(');start=text.index('{',start);end=start+1;depth=1
    while depth:
        depth+=(text[end]=='{')-(text[end]=='}');end+=1
    return text[start:end]

def assignment(code,name):
    return re.search(r'\b'+name+r'\s*=\s*([^;]+);',code).group(1)

def audit(text):
    functions=['quant_matmul_relu_quant_vector']
    if 'void QmqPaddedVector(' in text:functions.append('QmqPaddedVector')
    count=0
    max_rows=int(re.search(r'kVectorTileRows\s*=\s*(\d+)',text).group(1))
    for name in functions:
        code=body(text,name)
        cap=int(re.search(r'kPairCapacity\s*=\s*(\d+)',code).group(1))
        limit=Expression(assignment(code,'rowBatchLimit'))
        padded=Expression(assignment(code,'paddedBatch')) if 'const bool paddedBatch' in code else None
        stride=Expression(assignment(code,'batchRowStride')) if 'batchRowStride =' in code else None
        scalar_bytes=Expression(re.search(r'InitBuffer\(scalarMaxBuffer,\s*([^;]+)\);',code).group(1).replace('sizeof(float)','4'))
        allocation_env={'kPairCapacity':cap,'kFloatBlock':8,'alignedRows':(max_rows+7)//8*8}
        ub=0
        for allocation in re.findall(r'pipe.InitBuffer\(\w+,\s*([^;]+)\);',code):
            allocation=allocation.replace('sizeof(float)','4').replace('sizeof(int32_t)','4').replace('sizeof(int8_t)','1')
            parts=allocation.split(',')
            depth=Expression(parts[0]).value(allocation_env) if len(parts)==2 else 1
            size=Expression(parts[-1]).value(allocation_env)
            ub+=depth*((size+31)//32*32)
        # CANN9.0 Select modes1/2 require8KiB temporary UB on Atlas A3.
        # Count this once as reserved capacity, rather than per Select call.
        select_reserve=8192 if ('VSEL_TENSOR_TENSOR_MODE' in text or 'VSEL_TENSOR_SCALAR_MODE' in text) else 0
        assert ub+select_reserve<=196608,f'{name}: UB allocation {ub} plus Select API reserve {select_reserve} exceeds196608 bytes'
        for n in range(1,13001):
            # Specialized padded function is reachable only through its checked dispatch range.
            if name=='QmqPaddedVector' and not (32<=n<=3072 and n%32):continue
            env={'n':n,'kPairCapacity':cap,'kFloatBlock':8}
            is_padded=bool(padded.value(env)) if padded else False
            env['paddedBatch']=is_padded
            env['batchRowStride']=stride.value(env) if stride else n
            batch=limit.value(env)
            allocated=scalar_bytes.value(env)
            assert batch>=1
            assert batch*8*4<=allocated,f'{name}: N={n}, batch={batch}, scalarMax requires {batch*32} bytes, allocated={allocated}'
            # Per-row arithmetic addresses INT32/FP32 elements: 8 elements
            # align a row to 32 bytes. Packed INT8 output is cast/copied as ONE
            # continuous block, so it has no per-row UB address requirement.
            # Multi-block padded INT8 output still requires a 32-element stride.
            assert batch==1 or n%8==0 or is_padded,f'{name}: N={n} flat multi-row FP32 layout is not aligned'
            if batch>1 or is_padded:
                required_stride=32 if is_padded else 8
                assert env['batchRowStride']%required_stride==0,f'{name}: N={n} invalid row stride'
                if not is_padded:
                    assert n<=6144 and env['batchRowStride']==n,f'{name}: flat batch crosses N tiles or changes row stride'
                    assert ('DataCopyExtParams outputCopy{1, currentN * pairRows' in code or
                            'paddedBatch ? currentN : currentN * pairRows' in code),f'{name}: flat output is not one continuous block'
                assert env['batchRowStride']*batch<=cap,f'{name}: N={n} exceeds workspace/FP32 capacity'
            count+=1
    return count

R=Path(__file__).resolve().parent
base=(R/'v91_latest_control_20261012/kernel.asc').read_text(encoding='utf-8')
if len(sys.argv)>1:
    path=Path(sys.argv[1])/'kernel.asc'
    text=path.read_text(encoding='utf-8')
    checks=audit(text)
    for name in ('QmqDiv','QmqCorrectDiv','QmqQuantizeNearHalf','quant_matmul_relu_quant_cube'):
        assert body(text,name)==body(base,name),name+' changed'
    print(json.dumps({'source_shape_audit_pass':True,'N_checks':checks,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))
    sys.exit(0)
records=[]
for folder in sorted(R.glob('v*_20261012*')):
    source=folder/'kernel.asc'
    if not source.exists():continue
    text=source.read_text(encoding='utf-8')
    entry={'version':folder.name,'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    try:
        entry['N_checks']=audit(text)
        for name in ('QmqDiv','QmqCorrectDiv','QmqQuantizeNearHalf','quant_matmul_relu_quant_cube'):
            assert body(text,name)==body(base,name),name+' changed'
        entry['pass']=True
    except (AssertionError,ValueError,AttributeError,KeyError) as error:
        entry.update({'pass':False,'finding':str(error)})
    records.append(entry)
(R/'round5_shape_audit.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(records,ensure_ascii=False))
# Two real defects serve as regression checks; device success must not erase them.
failures={x['version'] for x in records if not x['pass']}
assert failures=={'v94_latest_padded_20261012','v96_smallN_batch8_20261012','v110_partition_activation_cache_20261012'},failures
