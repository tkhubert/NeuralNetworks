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
// 3. implement other optimization method (adadelta, rmsprop...)
// 4. implement stride for ConvLayers
// 5. speed up ConvLayers
// 6. internalize the hyperparameter search : can we back prop?

#include "NN.h"
#include "Data.h"
#include "NeuralNetwork.h"

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
    
    RLFunc RFunc;
    
    int    batchSize = 10;
    int    nbEpochs  = 5;
    
    float dropRateI = 0.001;
    float dropRate  = 0.005;
    float friction  = 0.9;
    vector<float> lRV     = {0.0075};//{0.001, 0.005, 0.0075, 0.01};//{0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15};
    vector<float> lambdaV = {4};//{0.1, 1, 3, 5};
    
    SMCostFunc SMCost;
    
    for (size_t i=0; i<lambdaV.size(); ++i)
    {
        for (size_t j=0; j<lRV.size(); ++j)
        {
            vector<unique_ptr<Layer>> layers;
            layers.emplace_back(make_unique<FCLayer>(iS , dropRateI, RFunc));
            layers.emplace_back(make_unique<FCLayer>(100, dropRate , RFunc));
            layers.emplace_back(make_unique<FCLayer>(100, dropRate , RFunc));
            //layers.emplace_back(make_unique<FCLayer>(100, dropRate, RFunc));
            layers.emplace_back(make_unique<FCLayer>(10 , 0.       , RFunc));
            
            NMOptimizer   Optim(lRV[j], friction, lambdaV[i], batchSize, nbEpochs, tS);
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
    
    int    batchSize = 20;
    int    nbEpochs  = 100;
    
    float friction  = 0.9;
    vector<float> lRV     =  {0.004};//0.001, 0.005, 0.0075, 0.01};//{0.005, 0.01, 0.02, 0.05};//{0.001,0.002, 0.005, 0.01, 0.02};//{0.0075};//{0.001, 0.005, 0.0075, 0.01};//{0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15};
    vector<float> lambdaV = {0};//{4};//{0.1, 1, 3, 5};
    
    SMCostFunc SMCost;

    for (size_t i=0; i<lambdaV.size(); ++i)
    {
        for (size_t j=0; j<lRV.size(); ++j)
        {
            vector<unique_ptr<Layer>> layers;
            layers.emplace_back(make_unique<ConvLayer>    (28, 28,  1, 0, 0, RFunc));
            layers.emplace_back(make_unique<ConvLayer>    (24, 24, 20, 5, 0, RFunc));
            layers.emplace_back(make_unique<ConvPoolLayer>(12, 12, 20, 2, 0, IFunc));
            layers.emplace_back(make_unique<ConvLayer>    ( 8,  8, 40, 5, 0, RFunc));
            layers.emplace_back(make_unique<ConvPoolLayer>( 4,  4, 40, 2, 0, IFunc));
            layers.emplace_back(make_unique<FCLayer>      (100, 0.         , RFunc));
            layers.emplace_back(make_unique<FCLayer>      (10 , 0.         , RFunc));
            
            NMOptimizer   Optim(lRV[j], friction, lambdaV[i], batchSize, nbEpochs, tS);
            NeuralNetwork CNN (SMCost, move(layers));
            CNN.train(data, Optim);
        }
    }
}
//
int main(int argc, const char * argv[])
{
    MLP();
    //CL();
    return 0;
}

