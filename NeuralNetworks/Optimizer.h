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
    Optimizer(float _alpha, float _friction, float _lambda, size_t _batchSize, size_t _nbEpochs, size_t trainSetSize) :
        alpha(_alpha/_batchSize),
        friction(_friction),
        lambda(_lambda*_batchSize/trainSetSize),
        batchSize(_batchSize),
        nbEpochs(_nbEpochs),
        alphaBase(_alpha),
        lambdaBase(_lambda)
    {};
    
    string getName() const
    {
        stringstream ss;
        ss << alphaBase << "_" << friction << "_" << lambdaBase << "_" << batchSize << "_" << nbEpochs;
        return ss.str();
    }
    
    float alpha , alphaBase;
    float friction;
    float lambda, lambdaBase;
    size_t batchSize;
    size_t nbEpochs;
};
    
}

#endif
