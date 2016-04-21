//
//  LinearAlgebra.h
//  NeuralNetworks
//
//  Created by Thomas Hubert on 08/10/2015.
//  Copyright © 2015 Thomas Hubert. All rights reserved.
//

#ifndef LinearAlgebra_h
#define LinearAlgebra_h

// Matrix Operations
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
            T tmp = 0.;
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
    vector<T> Bt(M*P);
    MatTrans  (B, &Bt[0], M, P);
    MatMultABt(A, &Bt[0], C, N, M, P);
}
//
//
template <typename T>
inline void MatMultABtC(const T* const A, const T* const B, T* C, size_t N, size_t M, size_t P) noexcept
{
    for (size_t i=0; i<N; ++i)
    {
        for (size_t j=0; j<P; ++j)
        {
            T tmp = 0.;
            for (size_t k=0; k<M; ++k)
                tmp+= A[i*M+k]*B[j*M+k];
                
            C[i*P+j] += tmp;
        }
    }
}
//
//
template <typename T>
inline void MatMultABC(const T* const A, const T* const B, T* C, size_t N, size_t M, size_t P) noexcept
{
    vector<T> Bt(M*P);
    MatTrans  (B, &Bt[0], M, P);
    MatMultABtC(A, &Bt[0], C, N, M, P);
}
//
}
    

// Convolution, Correlation Operations
namespace NN
{
    // Y = Y + W corr X
    template <typename T>
    inline void CorrMat(const T* const W, const T* const X, T* Y, size_t depth, size_t W1, size_t W2, size_t X1, size_t X2) noexcept
    {
        auto Y1=X1-W1+1;
        auto Y2=X2-W2+1;
        
        auto idx = 0;
        vector<T> XMat(Y1*Y2*W1*W2);
        for (size_t i=0; i<Y1; ++i)
            for (size_t j=0; j<Y2; ++j)
                for (size_t k1=0; k1<W1; ++k1)
                    for (size_t k2=0; k2<W2; ++k2)
                        XMat[idx++] = X[(i+k1)*X2+(j+k2)];
        
        MatMultABtC(W, &XMat[0], Y, depth, W1*W2, Y1*Y2);
    }
    
    // Y = Y + W conv X
    template <typename T>
    inline void ConvMat(const T* const W, const T* const X, T* Y, size_t depth, size_t W1, size_t W2, size_t X1, size_t X2) noexcept
    {
        auto Y1=X1-W1+1;
        auto Y2=X2-W2+1;
        
        auto idx = 0;
        vector<T> XMat(Y1*Y2*W1*W2);
        for (size_t i=0; i<Y1; ++i)
            for (size_t j=0; j<Y2; ++j)
                for (size_t k1=0; k1<W1; ++k1)
                    for (size_t k2=0; k2<W2; ++k2)
                        XMat[idx++] = X[(i+W1-1-k1)*X2+(j+W2-1-k2)];
                        
        MatMultABtC(W, &XMat[0], Y, depth, W1*W2, Y1*Y2);
    }
    
}
#endif /* LinearAlgebra_h */
