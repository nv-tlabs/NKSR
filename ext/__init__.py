from pathlib import Path
from torch.utils.cpp_extension import load


def p(rel_path):
    abs_path = Path(__file__).parent / rel_path
    return str(abs_path)


sdfgen = load(name='nksr_sdfgen',
              sources=[p('sdfgen/bind.cpp'),
                       p('common/kdtree_cuda.cu'),
                       p('sdfgen/sdf_from_points.cu')],
              extra_cflags=['-O2'],
              extra_cuda_cflags=['-O2', '-Xcompiler -fno-gnu-unique'])
