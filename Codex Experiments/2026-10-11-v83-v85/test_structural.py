"""Offline checks only: no claim of ASC compilation or device synchronization."""
from pathlib import Path
import json
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / '2026-10-11-v80-v82' / 'v82' / 'kernel.asc'

def function_body(text, name):
    start = text.index(name + '(')
    opening = text.index('{', start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[opening:end]

def audit_changed_scope(text,version):
    """Static artifact linkage, NOT hardware execution evidence."""
    if version in ('v83','v85'):
        body=function_body(text,'QmqEnqueueWorkspaceTile')
        for rule in ('count / paddedRows','(rowStride - naturalStride) / kFloatBlock',
                     'static_cast<uint8_t>(naturalStride - rowN)',
                     'Duplicate(input, static_cast<int32_t>(0), alignedCount)'):
            assert rule in body,rule
        vector=function_body(text,'quant_matmul_relu_quant_vector')
        assert 'paddedBatch ? pairRows : 1U' in vector
        assert 'paddedBatch ? currentN : currentN * pairRows' in vector
        assert vector.count('nextRows, paddedBatch ? alignedN : 0U')==2
        assert vector.count('QmqMinValue(rowBatchLimit, rows), paddedBatch ? alignedN : 0U')==2
        assert 'kPairCapacity / batchRowStride' in vector
        assert 'lane < pairCount && lane % alignedN < currentN' in vector
    if version in ('v84','v85'):
        host=function_body(text,'run_kernel')
        assert host.index('if (directTinyLaunch)')<host.index('PlatformAscendCManager::GetInstance')
        early=host[:host.index('PlatformAscendCManager::GetInstance')]
        assert 'm, n, k, launch.blockM, launch.groupCount, 1U)' in early
        assert 'QmqAcquireWorkspace' in early and 'aclrtSynchronizeStream(stream)' in early
        plan=function_body(text,'QmqPlanTinyLaunch')
        assert 'cores = std::min<uint32_t>(cores, m)' in plan
    if version=='v83':assert 'directTinyLaunch' not in text
    if version=='v84':assert 'paddedBatch' not in text

class StructuralChecks(unittest.TestCase):
    def test_source_linkage_rejects_reverted_candidates(self):
        base=BASE.read_text(encoding='utf-8')
        for version in ('v83','v84','v85'):
            with self.assertRaises((AssertionError,ValueError)):
                audit_changed_scope(base,version)
            audit_changed_scope((ROOT/version/'kernel.asc').read_text(encoding='utf-8'),version)

    def test_candidate_contracts(self):
        base = BASE.read_text(encoding='utf-8')
        for version in ('v83','v84','v85'):
            path = ROOT / version / 'kernel.asc'
            self.assertTrue(path.is_file(), f'{version}: implementation missing')
            text = path.read_text(encoding='utf-8')
            for name in ('QmqDiv','QmqCorrectDiv','QmqQuantizeNearHalf',
                         'quant_matmul_relu_quant_cube','quant_matmul_relu_quant_fused'):
                self.assertEqual(function_body(text,name),function_body(base,name),name)
            signature = lambda x: x[x.index('extern "C" void run_kernel('):x.index('{',x.index('extern "C" void run_kernel('))]
            self.assertEqual(signature(text),signature(base))
            self.assertEqual(text.count('extern "C" void run_kernel('),1)
            self.assertNotIn('TODO: implement',text)

    def test_padded_dma_and_quantization(self):
        # Catch treating n*rows as one flat DMA or confusing byte/block strides.
        rng = np.random.default_rng(724207)
        cases = 0
        for n in (65,71,95,127,255,511,513,999,1000,1023,1535,2047,2049,3071):
            stride = (n+31)//32*32
            limit = min(4,6144//stride)
            for rows in (1,2,3,4,63,64,65,67):
                integers = rng.integers(-2000000,2000001,(rows,n),dtype=np.int32)
                scales1 = rng.uniform(-2,2,rows).astype(np.float32)
                scales2 = rng.uniform(-2,2,n).astype(np.float32)
                integers[0]=0
                expected_activation = np.maximum((integers.astype(np.float32)*scales1[:,None])*scales2,0)
                expected_max = expected_activation.max(axis=1)
                expected_scale = np.where(expected_max>0,expected_max/np.float32(127),np.float32(1))
                expected_output = np.clip(np.rint(expected_activation/expected_scale[:,None]),0,127).astype(np.int8)
                actual = np.full((rows,n),-128,dtype=np.int8)
                for group in range(0,rows,64):
                    for offset in range(group,min(group+64,rows),limit):
                        count=min(limit,min(group+64,rows)-offset)
                        natural = (n+7)//8*8
                        dst_stride_blocks=(stride-natural)//8
                        destination=np.zeros(count*stride,dtype=np.int32)
                        for r in range(count):
                            dst_start=r*(natural+dst_stride_blocks*8)
                            destination[dst_start:dst_start+n]=integers[offset+r]
                        x2=np.zeros(stride,dtype=np.float32);x2[:n]=scales2
                        activation=np.maximum((destination.reshape(count,stride).astype(np.float32)*scales1[offset:offset+count,None])*x2,0)
                        maxs=activation.max(axis=1)
                        ys=np.where(maxs>0,maxs/np.float32(127),np.float32(1))
                        out=np.clip(np.rint(activation/ys[:,None]),0,127).astype(np.int8).ravel()
                        packed=np.concatenate([out[r*stride:r*stride+n] for r in range(count)])
                        actual[offset:offset+count]=packed.reshape(count,n)
                        np.testing.assert_array_equal(ys.view(np.uint32),expected_scale[offset:offset+count].view(np.uint32))
                        self.assertLessEqual(stride*count,6144)
                        self.assertLessEqual(natural-n,7)
                        self.assertEqual(stride%32,0)
                np.testing.assert_array_equal(actual,expected_output)
                cases+=1
        self.assertEqual(cases,112)

    def test_tiny_partition_and_resources(self):
        # Each row exactly once, even with cores=1 and odd blockM.
        for m in range(1,97):
            for cores in list(range(1,65))+[2**31-1,2**32-1]:
                effective=min(m,cores)
                block=min(m,max(4,(m+effective-1)//effective))
                groups=(m+block-1)//block
                self.assertLessEqual(groups,cores)
                seen=[]
                for vector in range(2*groups):
                    start=(vector//2)*block+(0 if vector%2==0 else (block+1)//2)
                    end=min(m,(vector//2)*block+((block+1)//2 if vector%2==0 else block))
                    seen.extend(range(start,end))
                self.assertEqual(sorted(seen),list(range(m)))
        for rows in range(1,49):
            for n in range(1,33):
                for k in (1,7,16,32,33,64,65,96,127,128):
                    if n*k>1024:continue
                    stride=(k+31)//32*32
                    an=(n+7)//8*8
                    qc=rows*32; cc=(qc+63)//64*64
                    align32=lambda x:(x+31)//32*32
                    sizes=[rows*stride,n*stride,rows*stride*2,n*stride*2,
                           rows*stride*4,n*stride*4,n*stride*4,2*an*4,
                           4*cc*4,align32(cc//8),qc*4,align32(rows*4),
                           align32(n*4),rows*32,align32(rows*4)]
                    self.assertLess(sum(map(align32,sizes)),196608)

    def test_prefetch_boundary_and_half_rounding(self):
        # Prefetch count may cross group64; only current group's rows are consumed.
        for limit in (1,2,3,4):
            for rows in range(1,1025):
                queued=[list(range(min(limit,rows)))]
                consumed=[]
                for group in range(0,rows,64):
                    group_rows=min(64,rows-group)
                    for local in range(0,group_rows,limit):
                        count=min(limit,group_rows-local)
                        start=group+local
                        payload=queued.pop(0)
                        self.assertEqual(payload[:count],list(range(start,start+count)))
                        consumed.extend(payload[:count])
                        if start+count<rows:
                            queued.append(list(range(start+count,min(start+count+limit,rows))))
                self.assertEqual(consumed,list(range(rows)))
                self.assertEqual(queued,[])
        # Padded lanes may be mask candidates but must never cross into a real row.
        for n in (65,95,999,2047,2049,3071):
            stride=(n+31)//32*32
            count=min(4,6144//stride)
            source=np.zeros((count,stride),np.float32)
            source[:,:n]=np.arange(n,dtype=np.float32)%127+np.float32(.5)
            source[:,0]=np.float32(.5)
            expected=np.clip(np.rint(source[:,:n]),0,127).astype(np.int8)
            self.assertTrue(np.all(expected[:,0]==0))
            compare_count=(count*stride+63)//64*64
            true_lanes=[i for i in range(compare_count) if i<count*stride and i%stride<n]
            self.assertEqual(len(true_lanes),count*n)
            masked=np.zeros_like(source,dtype=np.int8)
            masked.ravel()[true_lanes]=expected.ravel()
            np.testing.assert_array_equal(masked[:,:n],expected)

if __name__=='__main__':
    unittest.main(verbosity=2)
