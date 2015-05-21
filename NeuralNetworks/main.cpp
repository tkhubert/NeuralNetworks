//
//  main.cpp
//  NeuralNetworks
//
//  Created by Thomas Hubert on 07/04/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//


#include "includes.h"
#include "Data.h"
#include "NeuralNetwork.h"

int main(int argc, const char * argv[])
{
    std::string dir = "/Users/tkhubert/Documents/Projects/NeuralNetworks/MNist/";
    std::string trainLabels = dir + "train-labels-idx1-ubyte";
    std::string testLabels  = dir + "t10k-labels-idx1-ubyte";
    std::string trainData   = dir + "train-images-idx3-ubyte";
    std::string testData    = dir + "t10k-images-idx3-ubyte";

    MNistDataContainer data(trainLabels, testLabels, trainData, testData);
    size_t iS = data.getDataSize();
    
    SigmoidFunc SigFunc;
    RLFunc      RFunc(1, 0.1);
    MSECostFunc MSECFunc;
    CECostFunc  CECFunc;
    
    double learningRate = 0.25;
    double lambda       = 5;
    int    batchSize    = 10;
    int    nbEpochs     = 65;
    
    std::vector<Layer*> layers;
    FCLayer Layer0(0 , iS, SigFunc); layers.push_back(&Layer0);
    FCLayer Layer1(iS, 100, SigFunc); layers.push_back(&Layer1);
    FCLayer Layer2(100, 100, SigFunc); layers.push_back(&Layer2);
    FCLayer Layer3(100, 10, SigFunc); layers.push_back(&Layer3);
    
    Optimizer     Optim(learningRate, lambda, batchSize, nbEpochs, data.getTrainLabelData().size());
    NeuralNetwork FCNN(CECFunc, Optim, layers);
    FCNN.train(data);
    
    
//    std::vector<double> lambdaV = {0.25, 0.5, 1   , 3  , 5};
//    std::vector<double> lRV     = {0.05, 0.1, 0.25, 0.5, 1};
//    std::vector<CostFunc*> CFV;
//    CFV.push_back(&MSECFunc);
//    CFV.push_back(&CECFunc);
//    
//    for (size_t k=0; k<CFV.size(); ++k)
//    {
//        for (size_t i=0; i<lambdaV.size(); ++i)
//        {
//            for (size_t j=0; j<lRV.size(); ++j)
//            {
//                std::vector<Layer*> layers;
//                FCLayer Layer0(0 , iS, RFunc) ; layers.push_back(&Layer0);
//                FCLayer Layer1(iS , 100, RFunc); layers.push_back(&Layer1);
//                FCLayer Layer2(100, 100, RFunc); layers.push_back(&Layer2);
//                FCLayer Layer3(100, 10 , RFunc); layers.push_back(&Layer3);
//                
//                Optimizer     Optim(lRV[j], lambdaV[i], batchSize, nbEpochs, data.getTrainLabelData().size());
//                NeuralNetwork FCNN(*CFV[k], Optim, layers);
//                FCNN.train(data);
//            }
//        }
//    }
//    return 0;
}

