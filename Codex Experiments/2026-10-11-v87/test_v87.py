from pathlib import Path
import unittest
import importlib.util

ROOT=Path(__file__).resolve().parent
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
models=load(ROOT.parent/'2026-10-11-v83-v85/test_structural.py','dma_models')

class Composition(unittest.TestCase):
    def test_scopes(self):
        path=ROOT/'v87/kernel.asc'
        self.assertTrue(path.is_file(),'v87 candidate missing')
        text=path.read_text(encoding='utf-8')
        base=(ROOT.parent/'2026-10-11-v86/v86/kernel.asc').read_text(encoding='utf-8')
        dma=(ROOT.parent/'2026-10-11-v83-v85/v83/kernel.asc').read_text(encoding='utf-8')
        for name in ('QmqTryTinyDirect','QmqTinyVectorScaleSafe','qmq_tiny_vector_entry',
                     'run_kernel','QmqPlanTinyLaunch','QmqDiv','QmqCorrectDiv','QmqQuantizeNearHalf',
                     'quant_matmul_relu_quant_cube','quant_matmul_relu_quant_fused'):
            self.assertEqual(models.function_body(text,name),models.function_body(base,name),name)
        for name in ('QmqEnqueueWorkspaceTile','quant_matmul_relu_quant_vector'):
            self.assertEqual(models.function_body(text,name),models.function_body(dma,name),name)
        models.audit_changed_scope(text,'v85')

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Composition)
    for name in ('test_padded_dma_and_quantization','test_prefetch_boundary_and_half_rounding','test_tiny_partition_and_resources'):
        suite.addTest(models.StructuralChecks(name))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
