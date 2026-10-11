from pathlib import Path
import importlib.util,hashlib,json,difflib
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'2026-10-11-v86/v86/kernel.asc'
EXPECTED='13327d9537cb8b9f56c55db813b3b59793ab3d5eb5d406fcf77f27e94e0c347b'

def build():
    assert hashlib.sha256(BASE.read_bytes()).hexdigest()==EXPECTED
    module_path=ROOT.parent/'2026-10-11-v83-v85/build_candidates.py'
    spec=importlib.util.spec_from_file_location('dma_transform',module_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    base=BASE.read_text(encoding='utf-8')
    text='// V87: V86 tiny path plus reviewed padded general-row batching; NPU validation pending.\n'+module.padded_batch(base)
    folder=ROOT/'v87';folder.mkdir(exist_ok=True)
    source=folder/'kernel.asc';source.write_bytes(text.replace('\n','\r\n').encode('utf-8'))
    (folder/'vs_v86.diff').write_text(''.join(difflib.unified_diff(base.splitlines(True),text.splitlines(True),fromfile='v86',tofile='v87')),encoding='utf-8')
    report={'base_version':'v86','base_sha256':EXPECTED,
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'bytes':source.stat().st_size,'npu_status':'PENDING',
        'base_device_evidence':'User screenshot: 15/15 Pass, all error rates0%, submission ID unavailable.',
        'change':'General non32-aligned N64..3072 uses padded row batching; tiny functions/host/Cube unchanged.'}
    (ROOT/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':build()
