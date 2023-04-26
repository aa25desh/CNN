#include <iostream>
#include <cuda.h>
//you can change the grid_size
#define GRID_SIZE 128
//you can change the block_size
#define BLOCK_SIZE 128


#define TILE_WIDTH 8

__global__ void cnn(int N,int C,int K,int H,int W,int R, int S, int u, int v, int P, int Q,
               float *d_input, float * d_weight, float * d_output){

           int n = blockIdx.x;
           int k = blockIdx.z;
           int p = (blockIdx.y / (P/TILE_WIDTH + 1) )*TILE_WIDTH + threadIdx.y;
           int q = (blockIdx.y % (P/TILE_WIDTH + 1) )*TILE_WIDTH + threadIdx.x;
            
          //for(unsigned int n=0; n<N; n++) { // minibatch size
          //  for(unsigned int k=0; k<K; k ++) { // output feature map
            if (n < N && k < K && p < P && q < Q)
            {// input feature map
                float sum1 = 0;
                unsigned int ij1 = p * u; // input height
                unsigned int ii1 = q * v; // input width
                for(unsigned int c=0; c<C; c ++) { 
                    for (unsigned int r = 0; r<R; r ++) { // filter height
                        for (unsigned int s = 0; s < S; s ++) {// filter width
                            sum1 += d_input[n*C*H*W + c*H*W + (ij1+r)*W + ii1+s] * d_weight[k*C*R*S+c*R*S+r*S+s];
                            }
                        }
                    }
                d_output[n*K*P*Q + k*P*Q + p*Q + q] = sum1;
            }

}
/*
//N = 128, C = 832, K = 128, H = 7, W = 7, R = 1, S = 1, u = 1, v = 1, P = 7, Q = 7 
#define TILE_WIDTH 8
__global__ void cnn(int N,int C,int K,int H,int W,int R, int S, int u, int v, int P, int Q,
               float *d_input, float * d_weight, float * d_output){

           int n = blockIdx.x;
           int k = blockIdx.z;
           int p = (blockIdx.y / (P/TILE_WIDTH + 1) )*TILE_WIDTH + threadIdx.y;
           int q = (blockIdx.y % (P/TILE_WIDTH + 1) )*TILE_WIDTH + threadIdx.x;
            
          //for(unsigned int n=0; n<N; n++) { // minibatch size
          //  for(unsigned int k=0; k<K; k ++) { // output feature map
            if (n < N && k < K && p < P && q < Q)
            {// input feature map
                float sum1 = 0;
                unsigned int ij1 = p * u; // input height
                unsigned int ii1 = q * v; // input width
                for(unsigned int c=0; c<C; c=c+32) { 
                    //for (unsigned int r = 0; r<R; r ++) { // filter height
                    //    for (unsigned int s = 0; s < S; s ++) {// filter width
                        int _i = n*C*H*W + (ij1)*W + ii1;
                        int _w = k*C*R*S;
                        sum1 += d_input[ _i+ c*H*W] * d_weight[_w+c*R*S];
                        sum1 += d_input[ _i+ (c+1)*H*W] * d_weight[_w+(c+1)*R*S];
                        sum1 += d_input[ _i+ (c+2)*H*W] * d_weight[_w+(c+2)*R*S];
                        sum1 += d_input[ _i+ (c+3)*H*W] * d_weight[_w+(c+3)*R*S];
                        sum1 += d_input[ _i+ (c+4)*H*W] * d_weight[_w+(c+4)*R*S];
                        sum1 += d_input[ _i+ (c+5)*H*W] * d_weight[_w+(c+5)*R*S];
                        sum1 += d_input[ _i+ (c+6)*H*W] * d_weight[_w+(c+6)*R*S];
                        sum1 += d_input[ _i+ (c+7)*H*W] * d_weight[_w+(c+7)*R*S];
                        sum1 += d_input[ _i+ (c+8)*H*W] * d_weight[_w+(c+8)*R*S];
                        sum1 += d_input[ _i+ (c+9)*H*W] * d_weight[_w+(c+9)*R*S];
                        sum1 += d_input[ _i+ (c+10)*H*W] * d_weight[_w+(c+10)*R*S];
                        sum1 += d_input[ _i+ (c+11)*H*W] * d_weight[_w+(c+11)*R*S];
                        sum1 += d_input[ _i+ (c+12)*H*W] * d_weight[_w+(c+12)*R*S];
                        sum1 += d_input[ _i+ (c+13)*H*W] * d_weight[_w+(c+13)*R*S];
                        sum1 += d_input[ _i+ (c+14)*H*W] * d_weight[_w+(c+14)*R*S];
                        sum1 += d_input[ _i+ (c+15)*H*W] * d_weight[_w+(c+15)*R*S];
                        sum1 += d_input[ _i+ (c+16)*H*W] * d_weight[_w+(c+16)*R*S];
                        sum1 += d_input[ _i+ (c+17)*H*W] * d_weight[_w+(c+17)*R*S];
                        sum1 += d_input[ _i+ (c+18)*H*W] * d_weight[_w+(c+18)*R*S];
                        sum1 += d_input[ _i+ (c+19)*H*W] * d_weight[_w+(c+19)*R*S];
                        sum1 += d_input[ _i+ (c+20)*H*W] * d_weight[_w+(c+20)*R*S];
                        sum1 += d_input[ _i+ (c+21)*H*W] * d_weight[_w+(c+21)*R*S];
                        sum1 += d_input[ _i+ (c+22)*H*W] * d_weight[_w+(c+22)*R*S];
                        sum1 += d_input[ _i+ (c+23)*H*W] * d_weight[_w+(c+23)*R*S];
                        sum1 += d_input[ _i+ (c+24)*H*W] * d_weight[_w+(c+24)*R*S];
                        sum1 += d_input[ _i+ (c+25)*H*W] * d_weight[_w+(c+25)*R*S];
                        sum1 += d_input[ _i+ (c+26)*H*W] * d_weight[_w+(c+26)*R*S];
                        sum1 += d_input[ _i+ (c+27)*H*W] * d_weight[_w+(c+27)*R*S];
                        sum1 += d_input[ _i+ (c+28)*H*W] * d_weight[_w+(c+28)*R*S];
                        sum1 += d_input[ _i+ (c+29)*H*W] * d_weight[_w+(c+29)*R*S];
                        sum1 += d_input[ _i+ (c+30)*H*W] * d_weight[_w+(c+30)*R*S];
                        sum1 += d_input[ _i+ (c+31)*H*W] * d_weight[_w+(c+31)*R*S];


                            //}

                }
                        //}
                    //}
                d_output[n*K*P*Q + k*P*Q + p*Q + q] = sum1;
            }

}
/*
//N = 128, C = 3, K = 64, H = 112, W = 112, R = 3, S = 3, u = 2, v = 2, P = 55, Q = 55 
#define TILE_WIDTH 32
__global__ void unroll2_cnn(int N,int C,int K,int H,int W,int R, int S, int u, int v, int P, int Q,
               float *d_input, float * d_weight, float * d_output){

           int n = blockIdx.x;
           int k = blockIdx.z;
           int p = (blockIdx.y / (P/TILE_WIDTH + 1) )*TILE_WIDTH + threadIdx.y;
           int q = (blockIdx.y % (P/TILE_WIDTH + 1) )*TILE_WIDTH + threadIdx.x;
            
          //for(unsigned int n=0; n<N; n++) { // minibatch size
          //  for(unsigned int k=0; k<K; k ++) { // output feature map
            if (n < N && k < K && p < P && q < Q)
            {// input feature map
                float sum1 = 0;
                //for(unsigned int c=0; c<C; c ++) { 
                    //for (unsigned int r = 0; r<R; r ++) { // filter height
                    //    for (unsigned int s = 0; s < S; s ++) {// filter width
                            int _i = n*C*H*W + p*u*W + q*v;
                            int _w = k*C*R*S;
                            sum1 = sum1 + d_input[_i + 0*H*W + 0*W +0] * d_weight[_w+0*R*S+0*S+0] + 
                                          d_input[_i + 0*H*W + 0*W +1] * d_weight[_w+0*R*S+0*S+1] + 
                                          d_input[_i + 0*H*W + 0*W +2] * d_weight[_w+0*R*S+0*S+2] +
                                          d_input[_i + 0*H*W + 1*W +0] * d_weight[_w+0*R*S+1*S+0] + 
                                          d_input[_i + 0*H*W + 1*W +1] * d_weight[_w+0*R*S+1*S+1] + 
                                          d_input[_i + 0*H*W + 1*W +2] * d_weight[_w+0*R*S+1*S+2] +
                                          d_input[_i + 0*H*W + 2*W +0] * d_weight[_w+0*R*S+2*S+0] + 
                                          d_input[_i + 0*H*W + 2*W +1] * d_weight[_w+0*R*S+2*S+1] + 
                                          d_input[_i + 0*H*W + 2*W +2] * d_weight[_w+0*R*S+2*S+2] +
                                          d_input[_i + 1*H*W + 0*W +0] * d_weight[_w+1*R*S+0*S+0] +
                                            d_input[_i + 1*H*W + 0*W +1] * d_weight[_w+1*R*S+0*S+1] +
                                            d_input[_i + 1*H*W + 0*W +2] * d_weight[_w+1*R*S+0*S+2] +
                                            d_input[_i + 1*H*W + 1*W +0] * d_weight[_w+1*R*S+1*S+0] +
                                            d_input[_i + 1*H*W + 1*W +1] * d_weight[_w+1*R*S+1*S+1] +
                                            d_input[_i + 1*H*W + 1*W +2] * d_weight[_w+1*R*S+1*S+2] +
                                            d_input[_i + 1*H*W + 2*W +0] * d_weight[_w+1*R*S+2*S+0] +
                                            d_input[_i + 1*H*W + 2*W +1] * d_weight[_w+1*R*S+2*S+1] +
                                            d_input[_i + 1*H*W + 2*W +2] * d_weight[_w+1*R*S+2*S+2] +
                                            d_input[_i + 2*H*W + 0*W +0] * d_weight[_w+2*R*S+0*S+0] +
                                            d_input[_i + 2*H*W + 0*W +1] * d_weight[_w+2*R*S+0*S+1] +
                                            d_input[_i + 2*H*W + 0*W +2] * d_weight[_w+2*R*S+0*S+2] +
                                            d_input[_i + 2*H*W + 1*W +0] * d_weight[_w+2*R*S+1*S+0] +
                                            d_input[_i + 2*H*W + 1*W +1] * d_weight[_w+2*R*S+1*S+1] +
                                            d_input[_i + 2*H*W + 1*W +2] * d_weight[_w+2*R*S+1*S+2] +
                                            d_input[_i + 2*H*W + 2*W +0] * d_weight[_w+2*R*S+2*S+0] +
                                            d_input[_i + 2*H*W + 2*W +1] * d_weight[_w+2*R*S+2*S+1] +
                                            d_input[_i + 2*H*W + 2*W +2] * d_weight[_w+2*R*S+2*S+2];


                            //}
                        //}
                   // }
                d_output[n*K*P*Q + k*P*Q + p*Q + q] = sum1;
            }

}

*/

