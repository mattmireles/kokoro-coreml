// Direct expert-block read microbenchmark for macOS Stage 0 measurements.
//
// This program intentionally does one thing: read fixed-size expert blocks from
// a normal file with F_NOCACHE enabled and emit JSON timing data. The Python
// runner owns file creation, fs_usage capture, and gate decisions.

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static const size_t kPageAlignment = 16 * 1024;

typedef struct {
    int fd;
    size_t block_size;
    off_t offset;
    uint64_t latency_ns;
    ssize_t bytes_read;
    int error_code;
} ReadTask;

static uint64_t monotonic_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ((uint64_t)ts.tv_sec * 1000000000ull) + (uint64_t)ts.tv_nsec;
}

static void *read_task_main(void *arg) {
    ReadTask *task = (ReadTask *)arg;
    void *buffer = NULL;
    if (posix_memalign(&buffer, kPageAlignment, task->block_size) != 0) {
        task->error_code = ENOMEM;
        return NULL;
    }

    uint64_t start_ns = monotonic_ns();
    task->bytes_read = pread(task->fd, buffer, task->block_size, task->offset);
    task->latency_ns = monotonic_ns() - start_ns;
    if (task->bytes_read < 0) {
        task->error_code = errno;
    } else if ((size_t)task->bytes_read != task->block_size) {
        task->error_code = EIO;
    } else {
        task->error_code = 0;
    }
    free(buffer);
    return NULL;
}

static uint64_t next_random(uint64_t *state) {
    *state = (*state * 6364136223846793005ull) + 1442695040888963407ull;
    return *state;
}

static void usage(const char *program) {
    fprintf(stderr,
            "usage: %s --file PATH --block-size BYTES --iterations N "
            "--pattern sequential|random --queue-depth N [--start-delay-ms N]\n",
            program);
}

int main(int argc, char **argv) {
    const char *file_path = NULL;
    const char *pattern = "sequential";
    size_t block_size = 0;
    int iterations = 0;
    int queue_depth = 1;
    int start_delay_ms = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--file") == 0 && i + 1 < argc) {
            file_path = argv[++i];
        } else if (strcmp(argv[i], "--block-size") == 0 && i + 1 < argc) {
            block_size = (size_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--pattern") == 0 && i + 1 < argc) {
            pattern = argv[++i];
        } else if (strcmp(argv[i], "--queue-depth") == 0 && i + 1 < argc) {
            queue_depth = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--start-delay-ms") == 0 && i + 1 < argc) {
            start_delay_ms = atoi(argv[++i]);
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    if (file_path == NULL || block_size == 0 || iterations <= 0 || queue_depth <= 0) {
        usage(argv[0]);
        return 2;
    }
    if (strcmp(pattern, "sequential") != 0 && strcmp(pattern, "random") != 0) {
        usage(argv[0]);
        return 2;
    }

    int fd = open(file_path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "open failed: %s\n", strerror(errno));
        return 1;
    }

    int no_cache = 1;
    int f_nocache_result = fcntl(fd, F_NOCACHE, no_cache);
    int f_rdahead_result = -1;
#ifdef F_RDAHEAD
    int no_readahead = 0;
    f_rdahead_result = fcntl(fd, F_RDAHEAD, no_readahead);
#endif

    struct stat st;
    if (fstat(fd, &st) != 0) {
        fprintf(stderr, "fstat failed: %s\n", strerror(errno));
        close(fd);
        return 1;
    }
    if ((off_t)block_size > st.st_size) {
        fprintf(stderr, "block size exceeds file size\n");
        close(fd);
        return 1;
    }

    if (start_delay_ms > 0) {
        usleep((useconds_t)start_delay_ms * 1000);
    }

    int total_reads = iterations * queue_depth;
    ReadTask *tasks = calloc((size_t)total_reads, sizeof(ReadTask));
    uint64_t random_state = 0xC0DEC0FFEEull;
    off_t max_start = st.st_size - (off_t)block_size;
    uint64_t wall_start_ns = monotonic_ns();
    int failures = 0;

    for (int iter = 0; iter < iterations; iter++) {
        pthread_t *threads = calloc((size_t)queue_depth, sizeof(pthread_t));
        if (threads == NULL) {
            close(fd);
            free(tasks);
            return 1;
        }

        for (int q = 0; q < queue_depth; q++) {
            int index = (iter * queue_depth) + q;
            off_t offset = 0;
            if (strcmp(pattern, "random") == 0 && max_start > 0) {
                uint64_t r = next_random(&random_state);
                uint64_t slot_count = ((uint64_t)max_start / block_size) + 1;
                offset = (off_t)((r % slot_count) * block_size);
            } else if (max_start > 0) {
                uint64_t slot_count = ((uint64_t)max_start / block_size) + 1;
                offset = (off_t)(((uint64_t)index % slot_count) * block_size);
            }
            tasks[index].fd = fd;
            tasks[index].block_size = block_size;
            tasks[index].offset = offset;
            if (pthread_create(&threads[q], NULL, read_task_main, &tasks[index]) != 0) {
                tasks[index].error_code = errno;
                failures++;
            }
        }

        for (int q = 0; q < queue_depth; q++) {
            pthread_join(threads[q], NULL);
        }
        free(threads);
    }

    uint64_t wall_ns = monotonic_ns() - wall_start_ns;
    uint64_t total_bytes = 0;
    for (int i = 0; i < total_reads; i++) {
        if (tasks[i].error_code != 0) {
            failures++;
        } else {
            total_bytes += (uint64_t)tasks[i].bytes_read;
        }
    }

    printf("{\n");
    printf("  \"file\": \"%s\",\n", file_path);
    printf("  \"pattern\": \"%s\",\n", pattern);
    printf("  \"block_size\": %zu,\n", block_size);
    printf("  \"iterations\": %d,\n", iterations);
    printf("  \"queue_depth\": %d,\n", queue_depth);
    printf("  \"file_size\": %lld,\n", (long long)st.st_size);
    printf("  \"f_nocache_result\": %d,\n", f_nocache_result);
    printf("  \"f_rdahead_result\": %d,\n", f_rdahead_result);
    printf("  \"total_reads\": %d,\n", total_reads);
    printf("  \"successful_reads\": %d,\n", total_reads - failures);
    printf("  \"failed_reads\": %d,\n", failures);
    printf("  \"total_bytes_read\": %" PRIu64 ",\n", total_bytes);
    printf("  \"wall_time_ns\": %" PRIu64 ",\n", wall_ns);
    printf("  \"latencies_ns\": [");
    for (int i = 0; i < total_reads; i++) {
        printf("%s%" PRIu64, i == 0 ? "" : ", ", tasks[i].latency_ns);
    }
    printf("]\n");
    printf("}\n");

    free(tasks);
    close(fd);
    return failures == 0 ? 0 : 1;
}

