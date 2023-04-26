#include "driver.cu"

inline void chkerr(cudaError_t code)
{
    if (code != cudaSuccess)
    {
        std::cerr << "ERROR!!!:" << cudaGetErrorString(code) <<std::endl;
        exit(-1);
    }
}

int main(int argc, char *argv[]) {
    // READ PROBLEM SIZES
    if (argc != 10) exit(1);
    int N = atoi(argv[1]);  // minibatch size       128   128
    int C = atoi(argv[2]);  // input feature map    3     832
    int K = atoi(argv[3]);  // output feature map   64    128
    int H = atoi(argv[4]);  //                      112   7
    int W = atoi(argv[5]);  //                      112   7
    int R = atoi(argv[6]);  // filter height        3     1
    int S = atoi(argv[7]);  // filter width         3     1
    int u = atoi(argv[8]);  //                      2     1
    int v = atoi(argv[9]);  //                      2     1
    int P = (H-R)/u + 1;    // output height        56    7
    int Q = (W-S)/v + 1;    // output width         56    7
    printf("N = %d, C = %d, K = %d, H = %d, W = %d, R = %d, S = %d, u = %d, v = %d, P = %d, Q = %d \n", N, C, K, H, W, R, S, u, v, P, Q);
    int Z;
    float *output_seq = new float[N*K*P*Q];
    memset(output_seq,0, N * K * P * Q*sizeof(float));
    float *output_par = new float[N*K*P*Q];
    memset(output_par,0, N * K * P * Q*sizeof(float));
    float *input = new float[N*C*H*W];
    float *weight = new float[K*C*R*S];
    // ASSIGN INITIAL VALUES FOR INPUT AND WEIGHT

    for(unsigned int n=0; n<N; ++n){
        for(unsigned int c=0; c<C; ++c){
            for(unsigned int h=0; h<H; ++h){
                for(unsigned int w=0; w<W; ++w){
                    input[n*C*H*W + c*H*W + h*W + w] =  ((float)(n+c+h+w));
                }
            }
        }
    }
    for (unsigned int k=0; k<K; k++) {
        for (unsigned int c=0; c<C; c++) {
            for (unsigned int r =0; r<R; r++) {
                for (unsigned int s =0; s<S; s++) {
                    //weight[k][c][r][s] = ((float) (k+c+r+s));
                    weight[k*C*R*S + c*R*S + r*S + s] = ((float) (k+c+r+s));
                }
            }
        }
    }
    // TIME SEQUENTIAL CALCULATION
    cudaEvent_t seq_start,seq_stop;
    float seq_time;
    cudaEventCreate(&seq_start);
    cudaEventCreate(&seq_stop);
    cudaEventRecord(seq_start);

    for(unsigned int n=0; n<N; n++) { // minibatch size
        for(unsigned int k=0; k<K; k ++) { // output feature map
            for(unsigned int c=0; c<C; c ++) { // input feature map
                for(unsigned int p=0; p<P; p ++) { // output height
                    unsigned int ij = p * u; // input height
                    for (unsigned int q = 0; q<Q; q ++) { // output width
                        unsigned int ii = q * v; // input width
                        for (unsigned int r = 0; r<R; r ++) { // filter height
                            for (unsigned int s = 0; s < S; s ++) {// filter width
                                //output_seq[n][k][p][q] += input [n][c][ij+r][ii+s] * weight[k][c][r][s];
                                output_seq[n*K*P*Q + k*P*Q + p*Q + q] += input[n*C*H*W + c*H*W + (ij+r)*W + ii+s] * weight[k*C*R*S+c*R*S+r*S+s];
                            }
                        }
                    }
                }
            }
        }
    }

    cudaEventRecord(seq_stop);
    cudaEventSynchronize(seq_stop);
    cudaEventElapsedTime(&seq_time,seq_start, seq_stop);
    //@@ Copy input, weight and output data, input as example
    float * d_input, *d_weight, * d_output;
    chkerr(cudaMalloc((void **) &d_input,  sizeof(float) * N * C * H * W));
    chkerr(cudaMalloc((void **) &d_weight, sizeof(float) * K * C * R * S));
    chkerr(cudaMalloc((void **) &d_output, sizeof(float) * N * K * P * Q));

    chkerr(cudaMemcpy(d_input, input, sizeof(float) * N * C * H * W, cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(d_weight, weight, sizeof(float) * K * C * R * S, cudaMemcpyHostToDevice));

    // INITIALIZE PARALLEL TIMER
    cudaEvent_t par_start,par_stop;
    float par_time;
    cudaEventCreate(&par_start);
    cudaEventCreate(&par_stop);
    cudaEventRecord(par_start);

    //@@ Launch the GPU Kernel here, you may want multiple implementations to compare

    //cnn<<<GRID_SIZE, BLOCK_SIZE/8>>>(N,C,K,H,W,R,S,u,v,P,Q,d_input,d_weight,d_output);
    //printf("CPU sum = %f \n", output_seq[50]);
    int grid_width = ceil( P / (float) TILE_WIDTH);

    //Z = ceil( P / (float) TILE_WIDTH) * ceil(Q / (float) TILE_WIDTH);
    Z = ( P/TILE_WIDTH + 1) * (Q/TILE_WIDTH + 1);
    //printf("Z = %d, TILE_WIDTH = %d\n", Z, grid_width);
    
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(N, Z, K);
    cnn<<<gridDim, blockDim>>>(N,C,K,H,W,R,S,u,v,P,Q,d_input,d_weight,d_output);//, grid_width);

    cudaEventRecord(par_stop);
    cudaEventSynchronize(par_stop);
    cudaEventElapsedTime(&par_time,par_start, par_stop);

    //@@ Copy the GPU memory back to the CPU here
    chkerr(cudaMemcpy(output_par, d_output, sizeof(float) * N * K * P * Q, cudaMemcpyDeviceToHost));

    //@@ Free the GPU memory here
    cudaFree(d_input);

    // VERIFY CORRECTNESS BY COMPARING OUTPUTS
    for (unsigned int n=0; n<N; n++) { // minibatch size
        for (unsigned int k=0; k<K; k ++) { // output feature map
            for (unsigned int p=0; p<P; p ++) { // output height
                for (unsigned int q =0; q<Q; q ++) { // output width
                    if(abs(output_seq[n*K*P*Q+k*P*Q+p*Q+q]-output_par[n*K*P*Q+k*P*Q+p*Q+q])> .001) {
                        printf("%f \n" , abs(output_seq[n*K*P*Q+k*P*Q+p*Q+q]-output_par[n*K*P*Q+k*P*Q+p*Q+q]));  //here is the problem
                        printf("Outputs do not match!!!\n");
                        exit(2);
                    }
                }
            }
        }
    }

    // PRINT OUT SPEEDUP
    printf ("Sequential time = %f, Parallel time = %f, Speedup = %f\n",seq_time, par_time, seq_time/par_time);
}

