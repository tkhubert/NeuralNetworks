//
//  Optimizer.h
//  NeuralNetworks
//
//  Created by Thomas Hubert on 09/04/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//

#ifndef NeuralNetworks_Optimizer_h
#define NeuralNetworks_Optimizer_h

#include "NN.h"
namespace NN {
    
struct Optimizer
{
    Optimizer(float alpha, float friction, float lambda, size_t batchSize, size_t nbEpochs, size_t trainSetSize) :
        alpha(alpha/batchSize),
        friction(friction),
        lambda(lambda*batchSize/trainSetSize),
        batchSize(batchSize),
        nbEpochs(nbEpochs),
        alphaBase(alpha),
        lambdaBase(lambda)
    {};
    
    string getName() const
    {
        stringstream ss;
        ss << alphaBase << "_" << friction << "_" << lambdaBase << "_" << batchSize << "_" << nbEpochs;
        return ss.str();
    }
    
    float  alpha , alphaBase;
    float  friction;
    float  lambda, lambdaBase;
    size_t batchSize;
    size_t nbEpochs;
};
//
    
}

#endif
