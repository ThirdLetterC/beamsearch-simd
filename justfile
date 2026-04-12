set shell := ["bash", "-uc"]

cc := env_var_or_default("CC", "gcc")
nvcc := env_var_or_default("NVCC", "nvcc")
cflags := "-std=c2x -Wall -Wextra -Wpedantic -Werror -Iinclude -pthread"
cuda_cflags := cflags + " -DBEAMSEARCH_ENABLE_CUDA"
cuda_nvccflags := "-std=c++17 -Iinclude -Xcompiler=-Wall,-Wextra,-Werror"
test_src := "src/beamsearch.c tests/test_beamsearch.c"
bench_src := "src/beamsearch.c benchmark/run.c"

default:
    @just --list

compile:
    @{{cc}} @compile_flags.txt -O2 -c src/beamsearch.c -o /tmp/beamsearch.o
    @{{cc}} @compile_flags.txt -O2 -c tests/test_beamsearch.c -o /tmp/test_beamsearch.o

test:
    @{{cc}} {{cflags}} -O1 -g -fsanitize=address,undefined,leak {{test_src}} -lm -o /tmp/beamsearch_test
    @/tmp/beamsearch_test

test-avx2:
    @{{cc}} {{cflags}} -O3 -mavx2 -mfma {{test_src}} -lm -o /tmp/beamsearch_test_avx2
    @/tmp/beamsearch_test_avx2

psimd-check:
    @{{cc}} @compile_flags.txt -O2 -c include/psimd/test.c -o /tmp/psimd_test.o

check: compile test test-avx2 psimd-check

cuda-compile:
    @{{cc}} {{cuda_cflags}} -O2 -c src/beamsearch.c -o /tmp/beamsearch_cuda_cpu.o
    @{{nvcc}} {{cuda_nvccflags}} -O2 -c src/beamsearch_cuda.cu -o /tmp/beamsearch_cuda.o

cuda-test:
    @{{cc}} {{cuda_cflags}} -O2 -c src/beamsearch.c -o /tmp/beamsearch_cuda_cpu.o
    @{{cc}} {{cflags}} -O2 -c tests/test_beamsearch.c -o /tmp/test_beamsearch_cuda.o
    @{{nvcc}} {{cuda_nvccflags}} -O2 -c src/beamsearch_cuda.cu -o /tmp/beamsearch_cuda.o
    @{{nvcc}} /tmp/beamsearch_cuda_cpu.o /tmp/beamsearch_cuda.o /tmp/test_beamsearch_cuda.o -lm -lpthread -o /tmp/beamsearch_cuda_test
    @/tmp/beamsearch_cuda_test

check-cuda: cuda-compile cuda-test

bench scale="1":
    @{{cc}} {{cflags}} -O3 {{bench_src}} -lm -o /tmp/beamsearch_bench
    @/tmp/beamsearch_bench {{scale}}

bench-cuda scale="1":
    @{{cc}} {{cuda_cflags}} -O3 -c src/beamsearch.c -o /tmp/beamsearch_cuda_cpu.o
    @{{cc}} {{cflags}} -O3 -c benchmark/run.c -o /tmp/beamsearch_cuda_bench_main.o
    @{{nvcc}} {{cuda_nvccflags}} -O3 -c src/beamsearch_cuda.cu -o /tmp/beamsearch_cuda.o
    @{{nvcc}} /tmp/beamsearch_cuda_cpu.o /tmp/beamsearch_cuda.o /tmp/beamsearch_cuda_bench_main.o -lm -lpthread -o /tmp/beamsearch_cuda_bench
    @/tmp/beamsearch_cuda_bench {{scale}}

bench-avx2 scale="1":
    @{{cc}} {{cflags}} -O3 -mavx2 -mfma {{bench_src}} -lm -o /tmp/beamsearch_bench_avx2
    @/tmp/beamsearch_bench_avx2 {{scale}}

bench-compare scale="1": (bench scale) (bench-avx2 scale)
