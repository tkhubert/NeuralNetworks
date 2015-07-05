//
//  Optimizer.h
//  NeuralNetworks
//
//  Created by Thomas Hubert on 09/04/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//

#ifndef NeuralNetworks_Optimizer_h
#define NeuralNetworks_Optimizer_h

#include "includes.h"
struct Optimizer
{
    Optimizer(double _alpha, double _lambda, size_t _batchSize, size_t _nbEpochs, size_t trainSetSize) :
        alpha(_alpha/_batchSize),
        lambda(_lambda*_batchSize/trainSetSize),
        batchSize(_batchSize),
        nbEpochs(_nbEpochs),
        alphaBase(_alpha),
        lambdaBase(_lambda)
    {};
    
    std::string getName() const
    {
        std::stringstream ss;
        ss << alphaBase << "_" << lambdaBase << "_" << batchSize << "_" << nbEpochs;
        return ss.str();
    }
    
    double alpha , alphaBase;
    double lambda, lambdaBase;
    size_t batchSize;
    size_t nbEpochs;
};

#endif
