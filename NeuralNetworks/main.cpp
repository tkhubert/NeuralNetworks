//
//  main.cpp
//  NeuralNetworks
//
//  Created by Thomas Hubert on 07/04/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//

// TKH TODO
// 1. check the cost function and the check gradient
// 2. batch normalization.
// 3. try FFT to do the convolution.
// 4. internalize the hyperparameter search : can we back prop?

#include "NN.h"
#include "Data.h"
#include "NeuralNetwork.h"
#include "Layer.h"
#include "FCLayer.h"
#include "ConvLayer.h"
#include "ConvPoolLayer.h"

using namespace NN;

//
int main(int argc, const char * argv[])
{
    // data
    string dir = "/Users/tkhubert/Documents/Projects/NeuralNetworks/MNist/";
    string trainLabels = dir + "train-labels-idx1-ubyte";
    string testLabels  = dir + "t10k-labels-idx1-ubyte";
    string trainData   = dir + "train-images-idx3-ubyte";
    string testData    = dir + "t10k-images-idx3-ubyte";
    
    MNistDataContainer data(trainLabels, testLabels, trainData, testData);
    auto iS = data.getDataSize();
    auto tS = data.getTrainLabelData().size();
    
    // hyperparameter
    auto batchSize = 20;
    auto nbEpochs  = 2;
    auto friction  = 0.9;
    auto dropRateI = 0.05;
    auto dropRate  = 0.10;
    
    vec_r learnRates = {0.002};
    vec_r lambdas    = {0.1};//{0.1, 1, 3, 5};
    
    vector<pair_r> hyperParams;
    for (auto learnRate : learnRates)
        for (auto lambda : lambdas)
            hyperParams.push_back(make_pair(learnRate, lambda));
    
    // setting
    IdFunc     IFunc;
    RLFunc     RFunc;
    SMCostFunc SMCost;
    
    for (auto hyperParam : hyperParams)
    {
        auto learnRate = hyperParam.first;
        auto lambda    = hyperParam.second;
        
        L2Regularizer regularizer(lambda*batchSize/tS);
        NMOptimizer   optimizer(learnRate, friction);
        Trainer       trainer(optimizer, regularizer, batchSize, nbEpochs);
        
        if (true)
        {
            vector<unique_ptr<Layer>> layers;
            layers.emplace_back(make_unique<ConvLayer>    (28, 28, 1 , 0, 1, RFunc));
            layers.emplace_back(make_unique<ConvLayer>    (24, 24, 5 , 5, 1, RFunc));
            layers.emplace_back(make_unique<ConvPoolLayer>(12, 12, 5 , 2, 2, IFunc));
            layers.emplace_back(make_unique<ConvLayer>    ( 8,  8, 10, 5, 1, RFunc));
            layers.emplace_back(make_unique<ConvPoolLayer>( 4,  4, 10, 2, 2, IFunc));
            layers.emplace_back(make_unique<FCLayer>      (100, 0.         , RFunc));
            layers.emplace_back(make_unique<FCLayer>      (10 , 0.         , RFunc));
            NeuralNetwork CNN(SMCost, move(layers));
            
            CNN.train(data, trainer);
        }
        else
        {
            vector<unique_ptr<Layer>> layers;
            layers.emplace_back(make_unique<FCLayer>(iS , dropRateI, RFunc));
            layers.emplace_back(make_unique<FCLayer>(100, dropRate , RFunc));
            layers.emplace_back(make_unique<FCLayer>(100, dropRate , RFunc));
            layers.emplace_back(make_unique<FCLayer>(10 , 0.       , IFunc));
            NeuralNetwork FCNN(SMCost, move(layers));
            
            FCNN.train(data, trainer);
        }
        
    }
    
    return 0;
}

