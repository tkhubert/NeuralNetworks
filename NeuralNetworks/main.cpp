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
    size_t inputSize = data.getDataSize();
    
    SigmoidFunc AFunc;
    MSECostFunc CFunc;
    Optimizer   Optim(0.5, 0., 10, 30);
    
    FCLayer Layer0(0, inputSize, AFunc);
    FCLayer Layer1(inputSize, 100, AFunc);
    FCLayer Layer2(100, 10, AFunc);
    
    std::vector<Layer*> layers;
    layers.push_back(&Layer0);
    layers.push_back(&Layer1);
    layers.push_back(&Layer2);
    
    NeuralNetwork FCNN(CFunc, Optim, layers);
    FCNN.train(data);
    std::cout << FCNN.getCost() << ", " << FCNN.getErrRate() << std::endl;
    
    return 0;
}
