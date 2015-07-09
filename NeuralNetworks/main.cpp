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
// 3. use better than gradient descent method
// 4. internalize the hyperparameter search : can we back prop?

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
    RLFunc      RFunc;
    MSECostFunc MSECFunc;
    CECostFunc  CECFunc;
    SMCostFunc  SMCFunc;
    SVMCostFunc SVMCFunc;

    int    batchSize    = 10;
    int    nbEpochs     = 10;

    float friction = 0.9;
    std::vector<float> lRV     = {0.0075};//{0.001, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.03};//{0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15};
    std::vector<float> lambdaV = {2};//{0.1, 0.5, 1, 2, 3, 4, 5, 7, 10};
    std::vector<CostFunc*> CFV;
    
    //CFV.push_back(&MSECFunc);
    //CFV.push_back(&CECFunc);
    //CFV.push_back(&SVMCFunc);
    CFV.push_back(&SMCFunc);
    
    for (size_t k=0; k<CFV.size(); ++k)
    {
        for (size_t i=0; i<lambdaV.size(); ++i)
        {
            for (size_t j=0; j<lRV.size(); ++j)
            {
                std::vector<Layer*> layers;
                FCLayer Layer0(0  , iS , RFunc) ; layers.push_back(&Layer0);
                FCLayer Layer1(iS , 100, RFunc); layers.push_back(&Layer1);
                FCLayer Layer2(100, 100, RFunc); layers.push_back(&Layer2);
                //FCLayer Layer3(100, 100, RFunc); layers.push_back(&Layer3);
                FCLayer Layer4(100, 10 , RFunc); layers.push_back(&Layer4);
                
                Optimizer     Optim(lRV[j], friction, lambdaV[i], batchSize, nbEpochs, data.getTrainLabelData().size());
                NeuralNetwork FCNN(*CFV[k], Optim, layers);
                FCNN.train(data);
            }
        }
    }
    
    return 0;
}

