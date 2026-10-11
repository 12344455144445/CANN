"""Mechanical source transforms; originals never overwritten; inverse proof included."""
from pathlib import Path
import hashlib
import json
import difflib

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'2026-10-11-v80-v82'/'v82'/'kernel.asc'
BASE_HASH='849358aa5267ea24b600d6cb458db72abb2e282d6495ecd7b92de618ef062e37'

def replace(source,old,new,count=1):
    assert source.count(old)==count,(old[:110],source.count(old),count)
    result=source.replace(old,new)
    if new:
        assert result.replace(new,old)==source,'Inverse transformation failed'
    else:
        assert count == 1
        offset = source.index(old)
        assert result[:offset]+old+result[offset:]==source,'Deletion inverse failed'
    return result

def padded_batch(source):
    old='''    uint32_t count,
    uint32_t alignedCount)
{
    LocalTensor<int32_t> input = queue.AllocTensor<int32_t>();
    DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> pad{
        true, 0, static_cast<uint8_t>(alignedCount - count), 0};
    DataCopyPad(input, workspace[offset], copy, pad);
    queue.EnQue(input);
}'''
    new='''    uint32_t count,
    uint32_t alignedCount,
    uint32_t paddedRows = 1U,
    uint32_t rowStride = 0U)
{
    LocalTensor<int32_t> input = queue.AllocTensor<int32_t>();
    if (rowStride != 0U) {
        // DMA pads only to 8 int32 elements. Initialize the additional row gap.
        Duplicate(input, static_cast<int32_t>(0), alignedCount);
        PipeBarrier<PIPE_ALL>();
        const uint32_t rowN = count / paddedRows;
        const uint32_t naturalStride = QmqAlignUp(rowN, kFloatBlock);
        DataCopyExtParams copy{static_cast<uint16_t>(paddedRows),
            static_cast<uint32_t>(rowN * sizeof(int32_t)), 0,
            (rowStride - naturalStride) / kFloatBlock, 0};
        DataCopyPadExtParams<int32_t> pad{
            true, 0, static_cast<uint8_t>(naturalStride - rowN), 0};
        DataCopyPad(input, workspace[offset], copy, pad);
    } else {
        DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> pad{
            true, 0, static_cast<uint8_t>(alignedCount - count), 0};
        DataCopyPad(input, workspace[offset], copy, pad);
    }
    queue.EnQue(input);
}'''
    source=replace(source,old,new)
    old='''    const uint32_t rowBatchLimit = (n >= 64U && n <= 3072U && n % 32U == 0U)
        ? QmqMinValue(4U, kPairCapacity / n) : 1U;'''
    new='''    const bool paddedBatch = n >= 64U && n <= 3072U && n % 32U != 0U;
    const uint32_t batchRowStride = paddedBatch ? QmqAlignUp(n, 32U) : n;
    const uint32_t rowBatchLimit = (n >= 64U && n <= 3072U)
        ? QmqMinValue(4U, kPairCapacity / batchRowStride) : 1U;'''
    source=replace(source,old,new)
    source=replace(source,
        'const uint32_t alignedN = QmqAlignUp(currentN, kFloatBlock);',
        'const uint32_t alignedN = paddedBatch ? QmqAlignUp(currentN, 32U)\n            : QmqAlignUp(currentN, kFloatBlock);',2)
    source=replace(source,
        'static_cast<uint8_t>(alignedN - currentN), 0.0f};\n        DataCopyPad(x2ScaleLocal, x2ScaleGm[nOffset], x2ScaleCopy, x2ScalePad);',
        '''static_cast<uint8_t>(QmqAlignUp(currentN, kFloatBlock) - currentN), 0.0f};
        if (paddedBatch) {
            Duplicate(x2ScaleLocal, 0.0f, alignedN);
            PipeBarrier<PIPE_ALL>();
        }
        DataCopyPad(x2ScaleLocal, x2ScaleGm[nOffset], x2ScaleCopy, x2ScalePad);''',2)
    source=replace(source,
        'currentN * QmqMinValue(rowBatchLimit, rows), alignedN * QmqMinValue(rowBatchLimit, rows));',
        '''currentN * QmqMinValue(rowBatchLimit, rows), alignedN * QmqMinValue(rowBatchLimit, rows),
            QmqMinValue(rowBatchLimit, rows), paddedBatch ? alignedN : 0U);''',2)
    source=replace(source,
        'currentN * nextRows, alignedN * nextRows);',
        'currentN * nextRows, alignedN * nextRows, nextRows, paddedBatch ? alignedN : 0U);',2)
    source=replace(source,
        'DataCopyExtParams outputCopy{1, currentN * pairRows, 0, 0, 0};',
        '''// Each padded INT8 row has a 32-byte-aligned UB stride; GM rows stay compact.
                DataCopyExtParams outputCopy{
                    static_cast<uint16_t>(paddedBatch ? pairRows : 1U),
                    paddedBatch ? currentN : currentN * pairRows, 0, 0, 0};''')
    return source

TINY_PLAN='''struct QmqTinyLaunchPlan {
    uint32_t blockM;
    uint32_t groupCount;
};

inline QmqTinyLaunchPlan QmqPlanTinyLaunch(uint32_t m, uint32_t cores)
{
    cores = std::min<uint32_t>(cores, m);
    const uint32_t blockM = std::min<uint32_t>(m,
        std::max<uint32_t>(4U, QmqHostCeilDiv(m, cores)));
    return {blockM, QmqHostCeilDiv(m, blockM)};
}

'''

def tiny_launch(source):
    anchor='} // namespace\n\nextern "C" void run_kernel('
    source=replace(source,anchor,TINY_PLAN+anchor)
    definitions='''    const uint32_t m = static_cast<uint32_t>(m64);
    const uint32_t n = static_cast<uint32_t>(n64);
    const uint32_t k = static_cast<uint32_t>(k64);
    const uint32_t usableCoreNum = static_cast<uint32_t>(std::min<int64_t>(
        availableCoreNum, static_cast<int64_t>(UINT32_MAX)));
'''
    source=replace(source,definitions,'')
    anchor='    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();'
    early=definitions+'''    const bool directTinyLaunch = m <= kQmqTinyMaxM && n <= kQmqTinyMaxN &&
        k <= 128U && static_cast<uint64_t>(n) * k <= 1024U;
    if (directTinyLaunch) {
        const QmqTinyLaunchPlan launch = QmqPlanTinyLaunch(m, usableCoreNum);
        uint8_t* resultWorkspace = nullptr;
        uint8_t* unusedSystemWorkspace = nullptr;
        // Result workspace preserves the exact device fallback for exceptional scales.
        if (!QmqAcquireWorkspace(static_cast<size_t>(m) * n * sizeof(int32_t), 1U,
                                 resultWorkspace, unusedSystemWorkspace)) {
            return;
        }
        qmq_tiny_vector_entry<<<2U * launch.groupCount, 0, stream>>>(
            x1, x2, resultWorkspace, x1Scale, x2Scale, y, yScale,
            m, n, k, launch.blockM, launch.groupCount, 1U);
        (void)aclrtSynchronizeStream(stream);
        return;
    }

'''
    source=replace(source,anchor,early+anchor)
    # The original late tiny dispatch is retained as unreachable for that gate.
    # Keeping it leaves the entire general-path launch/protocol untouched.
    return source

def build():
    assert hashlib.sha256(BASE.read_bytes()).hexdigest()==BASE_HASH
    base=BASE.read_text(encoding='utf-8')
    variants={'v83':padded_batch(base),'v84':tiny_launch(base),
              'v85':tiny_launch(padded_batch(base))}
    manifest={'base':'v82 / platform submission724207 / 15/15',
              'base_sha256':BASE_HASH,'device_validation':'PENDING',
              'note':'CPU proofs and source scopes do not establish ASC compilation, synchronization or score.',
              'versions':{}}
    for version,text in variants.items():
        text='// STRUCTURAL '+version.upper()+' CANDIDATE: device evaluation pending.\n'+text
        folder=ROOT/version;folder.mkdir(exist_ok=True)
        path=folder/'kernel.asc'
        path.write_bytes(text.replace('\n','\r\n').encode('utf-8'))
        (folder/'vs_v82.diff').write_text(''.join(difflib.unified_diff(base.splitlines(True),text.splitlines(True),fromfile='v82',tofile=version)),encoding='utf-8')
        manifest['versions'][version]={'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (ROOT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=='__main__':build()
