//
//  Optimizer.h
//  NeuralNetworks
//
//  Created by Thomas Hubert on 09/04/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//

#ifndef NeuralNetworks_Optimizer_h
#define NeuralNetworks_Optimizer_h

struct Optimizer
{
    Optimizer(double _alpha, double _lambda, size_t _batchSize, size_t _nbEpochs) : alpha(_alpha), lambda(_lambda), batchSize(_batchSize), nbEpochs(_nbEpochs) {};
    
    double alpha;
    double lambda;
    size_t batchSize;
    size_t nbEpochs;
};

#endif
