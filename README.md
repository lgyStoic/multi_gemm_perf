### env
H100 PCIE
- triton
python3.13
- cute
python3.12 
- cute cpp
build cute_tutorial_wgmma_tma_sm90 demo from cutlass repo

### For test
- triton
```
conda activate triton
python quick_gemm_test.py
```
- cute
```
conda activate triton
python quick_gemm_test.py
```
### test shape
(512, 512, 512),
(1024, 1024, 1024),
(1536, 1536, 1536),
(2048, 2048, 2048),
(3072, 3072, 3072),
(4096, 4096, 4096),
(8192, 8192, 8192),
(16384, 16384, 16384),
(384, 384, 3072),
(64,64, 5120),
(64,64,8192),
(256,256, 5120),
(256,256, 8192),
(512,512, 5120),
(512,512, 8192),
(3968000, 64, 192),
(1280000, 64, 384)