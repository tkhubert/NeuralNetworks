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
template <typename T>
inline void MatTrans(const T* const A, T* At, size_t N, size_t M) noexcept
{
    for (size_t i=0; i<M; ++i)
        for (size_t j=0; j<N; ++j)
            At[i*N+j] = A[j*M+i];
}
//
// input A(N*M) and B(P*M), output C(N*P) = A * B^T
template <typename T>
inline void MatMultABt(const T* const A, const T* const B, T* C, size_t N, size_t M, size_t P) noexcept
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
template <typename T>
inline void MatMultAB(const T* const A, const T* const B, T* C, size_t N, size_t M, size_t P) noexcept
{
    vec_r Bt(M*P);
    MatTrans(B, &Bt[0], M, P);
    
    MatMultABt(A, &Bt[0], C, N, M, P);
}
//

    
}

#endif /* LinearAlgebra_h */
