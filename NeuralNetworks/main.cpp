//
//  main.cpp
//  NeuralNetworks
//
//  Created by Thomas Hubert on 07/04/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//

// TKH TO DO
// 1. center/normalize data
// 2. partition data properly
// 3. speed up ConvLayers: refine Img2Mat and try FFT to do the convolution.
// 4. internalize the hyperparameter search : can we back prop?

#include "NN.h"
#include "Data.h"
#include "NeuralNetwork.h"
#include "Layer.h"
#include "FCLayer.h"
#include "ConvLayer.h"
#include "ConvPoolLayer.h"

using namespace NN;

void MLP()
{
    string dir = "/Users/tkhubert/Documents/Projects/NeuralNetworks/MNist/";
    string trainLabels = dir + "train-labels-idx1-ubyte";
    string testLabels  = dir + "t10k-labels-idx1-ubyte";
    string trainData   = dir + "train-images-idx3-ubyte";
    string testData    = dir + "t10k-images-idx3-ubyte";
    
    MNistDataContainer data(trainLabels, testLabels, trainData, testData);
    auto iS = data.getDataSize();
    auto tS = data.getTrainLabelData().size();
    
    IdFunc IFunc;
    RLFunc RFunc;
    
    auto batchSize =  20;
    auto nbEpochs  =  2;
    auto dropRateI = 0.05;
    auto dropRate  = 0.10;
    auto friction  = 0.9;
    
    vec_r lR     = {0.005};//{0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15};
    vec_r lambda = {0.};

    SMCostFunc SMCost;
    
    for (auto lbda : lambda)
    {
        for (auto learningRate : lR)
        {
            vector<unique_ptr<Layer>> layers;
            layers.emplace_back(make_unique<FCLayer>(iS , dropRateI, RFunc));
            layers.emplace_back(make_unique<FCLayer>(100, dropRate , RFunc));
            layers.emplace_back(make_unique<FCLayer>(100, dropRate , RFunc));
            layers.emplace_back(make_unique<FCLayer>(10 , 0.       , IFunc));
            
            NMOptimizer   Optim(learningRate, friction, lbda, batchSize, nbEpochs, tS);
            NeuralNetwork FCNN (SMCost, move(layers));
            FCNN.train(data, Optim);
        }
    }
}
//
void CL()
{
    string dir = "/Users/tkhubert/Documents/Projects/NeuralNetworks/MNist/";
    string trainLabels = dir + "train-labels-idx1-ubyte";
    string testLabels  = dir + "t10k-labels-idx1-ubyte";
    string trainData   = dir + "train-images-idx3-ubyte";
    string testData    = dir + "t10k-images-idx3-ubyte";
    
    MNistDataContainer data(trainLabels, testLabels, trainData, testData);
    auto tS = data.getTrainLabelData().size();
    
    IdFunc IFunc;
    RLFunc RFunc;
    
    auto  batchSize = 20;
    auto  nbEpochs  = 2;
    auto  friction  = 0.9;
    vec_r lR        = {0.002};
    vec_r lambda    = {0.};//{0.1, 1, 3, 5};
    
    SMCostFunc SMCost;

    for (auto lbda : lambda)
    {
        for (auto learningRate : lR)
        {
            vector<unique_ptr<Layer>> layers;
            layers.emplace_back(make_unique<ConvLayer>    (28, 28, 1 , 0, 1, RFunc));
            layers.emplace_back(make_unique<ConvLayer>    (24, 24, 5 , 5, 1, RFunc));
            layers.emplace_back(make_unique<ConvPoolLayer>(12, 12, 5 , 2, 2, IFunc));
            layers.emplace_back(make_unique<ConvLayer>    ( 8,  8, 10, 5, 1, RFunc));
            layers.emplace_back(make_unique<ConvPoolLayer>( 4,  4, 10, 2, 2, IFunc));
            layers.emplace_back(make_unique<FCLayer>      (100, 0.         , RFunc));
            layers.emplace_back(make_unique<FCLayer>      (10 , 0.         , RFunc));
            
            NMOptimizer   Optim(learningRate, friction, lbda, batchSize, nbEpochs, tS);
            NeuralNetwork CNN (SMCost, move(layers));
            CNN.train(data, Optim);
        }
    }
}
//
int main(int argc, const char * argv[])
{
    //MLP();
    CL();
    return 0;
}

