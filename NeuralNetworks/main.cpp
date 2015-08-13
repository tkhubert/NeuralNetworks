//
//  main.cpp
//  NeuralNetworks
//
//  Created by Thomas Hubert on 07/04/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//

// TKH TO DO
// 1. center/normalize data
// 2. use dropout
// 3. internalize the hyperparameter search : can we back prop?

#include "NN.h"
#include "Data.h"
#include "NeuralNetwork.h"

using namespace NN;

int main(int argc, const char * argv[])
{
    string dir = "/Users/tkhubert/Documents/Projects/NeuralNetworks/MNist/";
    string trainLabels = dir + "train-labels-idx1-ubyte";
    string testLabels  = dir + "t10k-labels-idx1-ubyte";
    string trainData   = dir + "train-images-idx3-ubyte";
    string testData    = dir + "t10k-images-idx3-ubyte";

    MNistDataContainer data(trainLabels, testLabels, trainData, testData);
    auto iS = data.getDataSize();
    auto tS = data.getTrainLabelData().size();
    
    SigmoidFunc SigFunc;
    RLFunc      RFunc;

    int    batchSize = 10;
    int    nbEpochs  = 150;

    float dropRateI = 0.;
    float dropRate  = 0.;
    float friction  = 0.9;
    vector<float> lRV     = {0.0075};//{0.001, 0.005, 0.0075, 0.01};//{0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15};
    vector<float> lambdaV = {0.1, 1};//, 3, 5};
    
    vector<unique_ptr<CostFunc>> CFV;
    //CFV.emplace_back(make_unique<MSECostFunc>());
    //CFV.emplace_back(make_unique<CECostFunc>());
    //CFV.emplace_back(make_unique<SVMCostFunc>());
    CFV.emplace_back(make_unique<SMCostFunc>());
    
    for (size_t k=0; k<CFV.size(); ++k)
    {
        for (size_t i=0; i<lambdaV.size(); ++i)
        {
            for (size_t j=0; j<lRV.size(); ++j)
            {
                vector<unique_ptr<Layer>> layers;
                layers.emplace_back(make_unique<FCLayer>(0  , iS , dropRateI, RFunc));
                layers.emplace_back(make_unique<FCLayer>(iS , 1024, dropRate , RFunc));
                layers.emplace_back(make_unique<FCLayer>(1024, 1024, dropRate , RFunc));
                //layers.emplace_back(make_unique<FCLayer>(100, 100, dropRate, RFunc));
                layers.emplace_back(make_unique<FCLayer>(1024, 10 , 0.       , RFunc));
                
                Optimizer     Optim(lRV[j], friction, lambdaV[i], batchSize, nbEpochs, tS);
                NeuralNetwork FCNN (*CFV[k], Optim, move(layers));
                FCNN.train(data);
            }
        }
    }
    
    return 0;
}

