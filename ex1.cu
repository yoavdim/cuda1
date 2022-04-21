#include "ex1.h"

//#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

__device__ void prefix_sum(int arr[], int arr_size) {
    int tid = threadIdx.x;
    int increment;
    // trick: allow running multiple times (modulu) & skip threads (negative arr_size)
    // arr_size must be the same for all or __syncthreads will cause deadlock
    tid      = (arr_size > 0)? tid % arr_size : 0; 
    arr_size = (arr_size > 0)? arr_size       : -arr_size;

    for (int stride = 1; stride < arr_size; stride *= 2) {
        if (tid >= stride) {
            increment = arr[tid - stride];
        }
        __syncthreads();
        if (tid >= stride) {
            arr[tid] += increment;
        }
        __syncthreads();
    }
    return; 
}

/**
 * Perform interpolation on a single image
 *
 * @param maps 3D array ([TILES_COUNT][TILES_COUNT][256]) of    
 *             the tiles’ maps, in global memory.
 * @param in_img single input image, in global memory.
 * @param out_img single output buffer, in global memory.
 */
__device__ 
void interpolate_device(uchar* maps ,uchar *in_img, uchar* out_img);

/**
* map between a thread to its tile, total tile number is TILES_COUNT^2
*/
__global__ int get_tile_id(int index) {
    int line = index / IMG_WIDTH;
    int col  = index % IMG_WIDTH;
    line = line / TILE_HEIGHT; // round down
    col  = col / TILE_WIDTH; 
    return line * TILES_COUNT + col;
}


__global__ void process_image_kernel(uchar *all_in, uchar *all_out, uchar *maps) {
    __shared__ int histograms[IMG_TILES][256]; // shared memory, so each block allocating a different one for each image
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int tnum = blockDim.x;

    for (int i = tid; i < IMG_TILES*256; i += tnum) { // set zero
        ((int*) histograms)[i] = 0;
    }
    __syncthreads();

    for (int index = tid; index < IMG_SIZE; index += tnum) { // calc histograms
        atomicAdd(&(histograms[get_tile_id(index)][all_in[IMG_SIZE*bid + index]]), 1);
    }
    __syncthreads();

    // run prefix sum in each tile --- ASSUME: tnum  >= 256
    for (int run=0; run < (IMG_TILES/(tnum/256)+1); run++) { // enforce same amount of entries to prefix_sum
        int tile = (tid/256) + run*(tnum/256);
        if (tile >= IMG_TILES) 
            prefix_sum(NULL, -256);  // keep internal syncthread from blocking the rest
        else 
            prefix_sum(&(histograms[tile][0]), 256);
    }
    __syncthreads();

    // create map
    for (int i = tid; i < IMG_TILES*256; i += tnum) { 
        int cdf = ((int*) histograms)[i];
        map[MAP_SIZE*bid + i] = (uchar) ((((double)cdf)*255)/(TILE_WIDTH*TILE_HEIGHT)); // cast will round down
    }
    __syncthreads();
    
    interpolate_device(maps, all_in, all_out);
    return; 
}

/* Task serial context struct with necessary CPU / GPU pointers to process a single image */
struct task_serial_context {
    // TODO define task serial memory buffers
    uchar *d_image_in;
    uchar *d_image_out;
    uchar *d_maps; 
};

/* Allocate GPU memory for a single input image and a single output image.
 * 
 * Returns: allocated and initialized task_serial_context. */
struct task_serial_context *task_serial_init()
{
    auto context = new task_serial_context;

    cudaMalloc((void**)&(context->d_image_in),  IMG_SIZE);
    cudaMalloc((void**)&(context->d_image_out), IMG_SIZE);
    cudaMalloc((void**)&(context->d_maps), MAP_SIZE);
    //TODO: allocate GPU memory for a single input image, a single output image, and maps

    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void task_serial_process(struct task_serial_context *context, uchar *images_in, uchar *images_out)
{
    //TODO: in a for loop:
    //   1. copy the relevant image from images_in to the GPU memory you allocated
    //   2. invoke GPU kernel on this image
    //   3. copy output from GPU memory to relevant location in images_out_gpu_serial
    for (int i=0; i < N_IMAGES; i++) {
        cudaMemcpy(context->d_image_in + i*IMG_SIZE, images_in + i*IMG_SIZE, IMG_SIZE, cudaMemcpyHostToDevice);
        process_image_kernel<<<1,THREAD_NUM>>>(context->d_image_in, context->d_image_out, context->d_maps);  // 
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaGetLastError());
        cudaMemcpy(context->d_image_out + i*IMG_SIZE, images_out + i*IMG_SIZE, IMG_SIZE, cudaMemcpyDeviceToHost);
    }
}

/* Release allocated resources for the task-serial implementation. */
void task_serial_free(struct task_serial_context *context)
{
    //TODO: free resources allocated in task_serial_init
    cudaFree(context->d_image_in);
    cudaFree(context->d_image_out);
    cudaFree(context->d_maps);
    free(context);
}

/* Bulk GPU context struct with necessary CPU / GPU pointers to process all the images */
struct gpu_bulk_context {
    // TODO define bulk-GPU memory buffers
    uchar *d_image_in;
    uchar *d_image_out;
    uchar *d_maps;
};

/* Allocate GPU memory for all the input images, output images, and maps.
 * 
 * Returns: allocated and initialized gpu_bulk_context. */
struct gpu_bulk_context *gpu_bulk_init()
{
    auto context = new gpu_bulk_context;
    cudaMalloc((void**)&(context->d_image_in),  IMG_SIZE*N_IMAGES);
    cudaMalloc((void**)&(context->d_image_out), IMG_SIZE*N_IMAGES);
    cudaMalloc((void**)&(context->d_maps), MAP_SIZE*N_IMAGES);
    //TODO: allocate GPU memory for all the input images, output images, and maps

    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void gpu_bulk_process(struct gpu_bulk_context *context, uchar *images_in, uchar *images_out)
{
    //TODO: copy all input images from images_in to the GPU memory you allocated
    //TODO: invoke a kernel with N_IMAGES threadblocks, each working on a different image
    //TODO: copy output images from GPU memory to images_out
    cudaMemcpy(context->d_image_in, images_in, IMG_SIZE*N_IMAGES, cudaMemcpyHostToDevice);
    process_image_kernel<<<N_IMAGES,THREAD_NUM>>>(context->d_image_in, context->d_image_out, context->d_maps);  // 
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
    cudaMemcpy(context->d_image_out, images_out, IMG_SIZE*N_IMAGES, cudaMemcpyDeviceToHost);
}

/* Release allocated resources for the bulk GPU implementation. */
void gpu_bulk_free(struct gpu_bulk_context *context)
{
    //TODO: free resources allocated in gpu_bulk_init
    cudaFree(context->d_image_in);
    cudaFree(context->d_image_out);
    cudaFree(context->d_maps);
    free(context);
}
