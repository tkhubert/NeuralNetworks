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
    Optimizer(double _alpha, double _lambda, double _batchSize) : alpha(_alpha), lambda(_lambda), batchSize(_batchSize) {};
    
    double alpha;
    double lambda;
    double batchSize;
};

#endif
