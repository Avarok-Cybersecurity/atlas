// SPDX-License-Identifier: AGPL-3.0-only
//
// PLE n-gram fault-in micro-benchmark — is 16 workers the knee?
//
// Replicates `spark-storage/src/ngram_cache_fault.rs` exactly: O_DIRECT
// positional reads of BLOCK=4096 bytes at random 4 KiB-aligned offsets in the
// real n-gram safetensors, one thread per worker calling pread (the engine
// uses `read_at`, the same syscall), each into its own aligned bounce buffer.
// Sweeps the worker count so the queue-depth curve is measured on THIS drive
// rather than assumed — `MAX_WORKERS = 16` ships with the comment "NVMe queue
// depth benefits flatten out well below this", which is an assumption.
//
// Reported: wall time, per-read latency (wall * workers / reads, i.e. the
// latency each worker actually sees) and aggregate IOPS / MiB/s.
//
// Build: gcc -O2 -pthread -o ple_fault_bench ple_fault_bench.c
// Run:   ./ple_fault_bench <file> [reads_per_sweep] [blocks_per_read]
//        Defaults: 32768 reads (the ~31.7K misses of a real 8K prefill), 1 block.
// O_DIRECT means the page cache is bypassed, so repeated sweeps are honest.
#define _GNU_SOURCE
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>

#define BLOCK 4096ULL

static const int WORKER_SWEEP[] = {1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128};
static const int SWEEP_N = sizeof(WORKER_SWEEP) / sizeof(WORKER_SWEEP[0]);

typedef struct {
    int fd;
    uint64_t file_blocks;   // number of whole BLOCKs in the file
    uint64_t reads;         // reads this worker performs
    uint64_t seed;
    size_t bytes_per_read;
    uint64_t done;
    int err;
} job_t;

// Same cheap hash the storage bench uses to spread offsets deterministically.
static inline uint64_t rand_block(uint64_t i, uint64_t seed, uint64_t nblocks) {
    uint64_t x = (i + seed) * 2654435761ULL;
    x ^= x >> 29;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 32;
    return x % (nblocks ? nblocks : 1);
}

static void *worker(void *arg) {
    job_t *j = (job_t *)arg;
    void *buf = NULL;
    if (posix_memalign(&buf, BLOCK, j->bytes_per_read) != 0) { j->err = 1; return NULL; }
    for (uint64_t i = 0; i < j->reads; i++) {
        uint64_t blk = rand_block(i, j->seed, j->file_blocks - (j->bytes_per_read / BLOCK));
        ssize_t n = pread(j->fd, buf, j->bytes_per_read, (off_t)(blk * BLOCK));
        if (n < 0) { j->err = errno; break; }
        j->done++;
    }
    free(buf);
    return NULL;
}




static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <file> [reads] [blocks_per_read]\n", argv[0]);
        return 2;
    }
    const char *path = argv[1];
    uint64_t reads = (argc > 2) ? strtoull(argv[2], NULL, 10) : 32768;
    size_t nblk = (argc > 3) ? (size_t)strtoul(argv[3], NULL, 10) : 1;
    size_t bytes = nblk * BLOCK;

    int fd = open(path, O_RDONLY | O_DIRECT);
    if (fd < 0) { perror("open O_DIRECT"); return 1; }
    struct stat st;
    if (fstat(fd, &st) != 0) { perror("fstat"); return 1; }
    uint64_t file_blocks = (uint64_t)st.st_size / BLOCK;

    printf("FILE %s  size %.1f GiB  blocks %llu  read %zu B (%zu x 4 KiB) O_DIRECT random\n",
           path, st.st_size / (1024.0 * 1024 * 1024), (unsigned long long)file_blocks, bytes, nblk);
    printf("reads/sweep %llu  (a real ~8K prefill missed ~31.7K rows)\n\n",
           (unsigned long long)reads);
    printf("%8s %10s %14s %12s %12s\n", "workers", "wall_ms", "per_read_us", "IOPS", "MiB/s");

    for (int s = 0; s < SWEEP_N; s++) {
        int w = WORKER_SWEEP[s];
        pthread_t *th = calloc(w, sizeof(pthread_t));
        job_t *jobs = calloc(w, sizeof(job_t));
        uint64_t per = reads / (uint64_t)w;
        if (per == 0) per = 1;
        double t0 = now_s();
        for (int i = 0; i < w; i++) {
            jobs[i].fd = fd;
            jobs[i].file_blocks = file_blocks;
            jobs[i].reads = per;
            jobs[i].seed = 0x9e3779b97f4a7c15ULL * (uint64_t)(i + 1) + (uint64_t)s * 1315423911ULL;
            jobs[i].bytes_per_read = bytes;
            pthread_create(&th[i], NULL, worker, &jobs[i]);
        }
        uint64_t done = 0;
        int err = 0;
        for (int i = 0; i < w; i++) {
            pthread_join(th[i], NULL);
            done += jobs[i].done;
            if (jobs[i].err) err = jobs[i].err;
        }
        double dt = now_s() - t0;
        if (err) { fprintf(stderr, "  workers=%d: read error %d\n", w, err); }
        double per_read_us = dt * 1e6 * (double)w / (double)(done ? done : 1);
        double iops = done / dt;
        printf("%8d %10.1f %14.1f %12.0f %12.1f\n",
               w, dt * 1e3, per_read_us, iops, iops * bytes / (1024.0 * 1024.0));
        free(th); free(jobs);
    }
    close(fd);
    return 0;
}
