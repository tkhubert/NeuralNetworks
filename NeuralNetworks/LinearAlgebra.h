//
//  LinearAlgebra.h
//  NeuralNetworks
//
//  Created by Thomas Hubert on 08/10/2015.
//  Copyright © 2015 Thomas Hubert. All rights reserved.
//

#ifndef LinearAlgebra_h
#define LinearAlgebra_h

namespace NN
{

// input A, output At = A^T
void MatTrans(const vector<float>& A, vector<float>& At, size_t N, size_t M)
{
    for (size_t i=0; i<M; ++i)
        for (size_t j=0; j<N; ++j)
            At[i*N+j] = A[j*M+i];
}
//
// input A(N*M) and B(P*M), output C(N*P) = A * B^T
void MatMultABt(const vector<float>& A, const vector<float>& B, vector<float>& C, size_t N, size_t M, size_t P)
{
    for (size_t i=0; i<N; ++i)
    {
        for (size_t j=0; j<P; ++j)
        {
            float tmp = 0.;
            for (size_t k=0; k<M; ++k)
                tmp+= A[i*M+k]*B[j*M+k];
            
            C[i*P+j] = tmp;
        }
    }
}
//
//  input A(N*M) and B(M*P), output C(N*P) = A * B = A * (B^T)^T = MatMultABt(A, B^T)
void MatMultAB(const vector<float>& A, const vector<float>& B, vector<float>& C, size_t N, size_t M, size_t P)
{
    vector<float> Bt(B.size());
    MatTrans(B, Bt, M, P);
    
    MatMultABt(A, Bt, C, N, M, P);
}
//

    
}

#endif /* LinearAlgebra_h */
