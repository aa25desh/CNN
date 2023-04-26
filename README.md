
## Strategies given in the question:

1) Applying parallelization strategies (block and thread decompositions) that consider data reuse (temporal and spatial), particularly to achieve global memory coalescing
2) Using loop permutation to change the memory access order
3) Using tiling and copying into shared memory to exploit reuse across threads 
4) Using shared memory to reorganize data layout
5) Padding shared memory data to avoid shared memory bank conflicts, as in transpose
6) Unroll or Unroll-and-jam and scalar replacement

## Data Access:

input[n*C*H*W + c*H*W + h*W + w] = input[n][c][h][w]
weight[k*C*R*S + c*R*S + r*S + s] = weight[k][c][r][s]
output_seq[n*K*P*Q + k*P*Q + p*Q + q] = output_seq[n][k][p][q]  P = (H-R)/u + 1;  int Q = (W-S)/v + 1;  

## Main Optimization: 

I have tried many ways to optimize the solution; the best approach 
i got for the method mentioned in the reference book in chapter 16.

Mapping the 4D array to 3D grid.

1) X dim -> N
2) Z dim -> K
3) Y dim -> P, Q; as P and Q are small so we can compress them into one dim.

Tile Size:

I have use 32, 16, 8 for tile size and got the best results for 8 overall.


[u1369232@kp362:cnn_assignment]$  ./cnn-gpu 1 3 64 112 112 3 3 2 2
N = 1, C = 3, K = 64, H = 112, W = 112, R = 3, S = 3, u = 2, v = 2, P = 55, Q = 55 
Sequential time = 44.835968, Parallel time = 0.051488, Speedup = 870.804199

[u1369232@kp362:cnn_assignment]$ ./cnn-gpu 128 3 64 112 112 3 3 2 2
N = 128, C = 3, K = 64, H = 112, W = 112, R = 3, S = 3, u = 2, v = 2, P = 55, Q = 55 
Sequential time = 5752.332520, Parallel time = 7.202400, Speedup = 798.668823


[u1369232@kp362:cnn_assignment]$ ./cnn-gpu 1 832 128 7 7 1 1 1 1
N = 1, C = 832, K = 128, H = 7, W = 7, R = 1, S = 1, u = 1, v = 1, P = 7, Q = 7 
Sequential time = 58.108768, Parallel time = 0.302752, Speedup = 191.935211

[u1369232@kp362:cnn_assignment]$ ./cnn-gpu 128 832 128 7 7 1 1 1 1
N = 128, C = 832, K = 128, H = 7, W = 7, R = 1, S = 1, u = 1, v = 1, P = 7, Q = 7 
Sequential time = 7429.266602, Parallel time = 9.868544, Speedup = 752.822998

## Loop Unrolling:

please see the kernel unroll1_cnn in the comments.
I did loop unrolling for the filter loops and got better results for input (128/1 832 128 7 7 1 1 1 1)

int _i = n*C*H*W + (ij1)*W + ii1;
int _w = k*C*R*S;
sum =  sum + d_input[ _i+ c*H*W] * d_weight[_w+c*R*S] + 
        d_input[ _i+ (c+1)*H*W] * d_weight[_w+(c+1)*R*S]...

        d_input[ _i+ (c+31)*H*W] * d_weight[_w+(c+31)*R*S];

[u1369232@kp361:cnn_assignment]$ ./cnn-gpu 1 832 128 7 7 1 1 1 1
N = 1, C = 832, K = 128, H = 7, W = 7, R = 1, S = 1, u = 1, v = 1, P = 7, Q = 7 
Sequential time = 58.815266, Parallel time = 0.053376, Speedup = 1101.904663

[u1369232@kp361:cnn_assignment]$ ./cnn-gpu 128 832 128 7 7 1 1 1 1
N = 128, C = 832, K = 128, H = 7, W = 7, R = 1, S = 1, u = 1, v = 1, P = 7, Q = 7 
Sequential time = 7489.583984, Parallel time = 7.545504, Speedup = 992.588928

## Loop permutation:

The code was pretty flexible to do loop permutation, but results don't change until i do loop unrolling.

## Shared memory:

I think shared memory could be used while doing filter loops (c,r,s) but i am short of time to implement it.

Other that this i have tried many other ways to optimize the code but i am not getting any better results.