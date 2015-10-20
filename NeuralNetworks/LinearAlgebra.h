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
inline void MatTrans(const vec_r& A, vec_r& At, size_t N, size_t M)
{
    for (size_t i=0; i<M; ++i)
        for (size_t j=0; j<N; ++j)
            At[i*N+j] = A[j*M+i];
}
//
// input A(N*M) and B(P*M), output C(N*P) = A * B^T
inline void MatMultABt(const vec_r& A, const vec_r& B, vec_r& C, size_t N, size_t M, size_t P)
{
    for (size_t i=0; i<N; ++i)
    {
        for (size_t j=0; j<P; ++j)
        {
            real tmp = 0.;
            for (size_t k=0; k<M; ++k)
                tmp+= A[i*M+k]*B[j*M+k];
            
            C[i*P+j] = tmp;
        }
    }
}
//
//  input A(N*M) and B(M*P), output C(N*P) = A * B = A * (B^T)^T = MatMultABt(A, B^T)
inline void MatMultAB(const vec_r& A, const vec_r& B, vec_r& C, size_t N, size_t M, size_t P)
{
    vec_r Bt(B.size());
    MatTrans(B, Bt, M, P);
    
    MatMultABt(A, Bt, C, N, M, P);
}
//

    
}

#endif /* LinearAlgebra_h */
