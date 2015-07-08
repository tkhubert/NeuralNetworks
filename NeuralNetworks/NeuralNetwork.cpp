//
//  NeuralNetwork.cpp
//  NeuralNetworks
//
//  Created by Thomas Hubert on 08/04/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//

#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(const CostFunc& _CFunc, const Optimizer& _Optim, std::vector<Layer*>& _layers) : nbLayers(0) , CFunc(_CFunc), Optim(_Optim)
{
    nbLayers = _layers.size();
    layers.resize(nbLayers);
    
    layers[0] = _layers[0];
    for (size_t i=1; i<nbLayers; ++i)
    {
        Layer* pLayer = layers[i-1];
        Layer* cLayer = _layers[i];
        assert(pLayer->getOutputSize()==cLayer->getInputSize());
        
        pLayer->setNextLayer(cLayer);
        cLayer->setPrevLayer(pLayer);
        layers[i] = cLayer;
    }
    
    inputSize  = layers[0]->getOutputSize();
    outputSize = layers[nbLayers-1]->getOutputSize();
    
    debugFile  = std::ofstream("/Users/tkhubert/Documents/Projects/NeuralNetworks/MNist/"+getName());
}
//
std::string NeuralNetwork::getName() const
{
    std::stringstream ss;
    for (size_t i=0; i<nbLayers; ++i)
        ss << layers[i]->getOutputSize() << "_";

    ss << CFunc.getName() << "_" << layers[nbLayers-1]->getAFunc().getName() << "_" << Optim.getName();
    ss << ".csv";

    return ss.str();
}
//
void NeuralNetwork::setNbData(size_t nbData)
{
    for (size_t i=0; i<nbLayers; ++i)
        layers[i]->setNbData(nbData);
}
//
void NeuralNetwork::initParams()
{
    for (size_t i=1; i<nbLayers; ++i)
        layers[i]->initParams();
}
//
void NeuralNetwork::updateParams()
{
    for (size_t i=1; i<nbLayers; ++i)
        layers[i]->updateParams(Optim.alpha, Optim.friction, Optim.lambda);
}
//
void NeuralNetwork::setInput(const LabelData& lD)
{
    setNbData(1);
    layers[0]->setA(lD.data);
}
//
void NeuralNetwork::setInput(std::vector<LabelData>::const_iterator start, std::vector<LabelData>::const_iterator end)
{
    size_t nbData   = std::distance(start, end);
    size_t dataSize = start->data.size();
    std::vector<double> input(nbData*dataSize);
    
    for (size_t b=0; b<nbData; ++b)
    {
        for (size_t i=0; i<dataSize; ++i)
        {
            input[i*nbData+b] = (start+b)->data[i];
        }
    }
    
    setNbData(nbData);
    layers[0]->setA(input);
}
//
void NeuralNetwork::fwdProp(const LabelData& lD)
{
    setInput(lD);
    for (size_t i=1; i<nbLayers; ++i)
        layers[i]->fwdProp();
}
//
void NeuralNetwork::fwdProp(std::vector<LabelData>::const_iterator start, std::vector<LabelData>::const_iterator end)
{
    setInput(start, end);
    for (size_t i=1; i<nbLayers; ++i)
        layers[i]->fwdProp();
}
//
void NeuralNetwork::bwdProp(const std::vector<double>& dC)
{
    setDCost(dC);
    for (size_t i=nbLayers-1; i>=1; --i)
        layers[i]->bwdProp();
}
//
double NeuralNetwork::calcCost(std::vector<LabelData>::const_iterator start, std::vector<LabelData>::const_iterator end) const
{
    return CFunc.f(getOutput(), start, end);
}
//
void NeuralNetwork::calcDCost(std::vector<LabelData>::const_iterator start, std::vector<LabelData>::const_iterator end, std::vector<double>& dC)
{
    return CFunc.df(getOutput(), start, end, dC);
}
//
const std::vector<double>& NeuralNetwork::predict(const LabelData& lD)
{
    fwdProp(lD);
    return getOutput();
}
//
size_t NeuralNetwork::isCorrect(std::vector<LabelData>::const_iterator start, std::vector<LabelData>::const_iterator end) const
{
    const std::vector<double>& prediction = getOutput();
    size_t nbData = std::distance(start, end);
    
    size_t nbCorrect = 0;
    for (size_t b=0; b<nbData; ++b)
    {
        size_t maxIdx = 0;
        double runningMax = prediction[b];
        
        for (size_t i=0; i<outputSize; ++i)
        {
            double val = prediction[i*nbData+b];
            if (val>runningMax)
            {
                val    = runningMax;
                maxIdx = i;
            }
        }
        nbCorrect += (maxIdx==(start+b)->label);
    }
    return nbCorrect;
}
//
void NeuralNetwork::train(const DataContainer& data)
{
    std::vector<LabelData> lData = data.getTrainLabelData();
    
    std::cout << "Start training " << getName() << "-------------"<<std::endl;
    
    size_t totalISize = lData.size();
    size_t nbBatches  = (totalISize-1)/Optim.batchSize + 1;
    
    for (size_t t=0; t<Optim.nbEpochs; ++t)
    {
        debugFile << t << ", ";
        std::cout << "Epoch: " << t << ", ";
        std::clock_t startTimeEpoch = std::clock();
        
        std::random_shuffle(lData.begin(), lData.end());
        
        for (size_t batch=0; batch<nbBatches; ++batch)
        {
            size_t start  = batch*Optim.batchSize;
            size_t end    = std::min(start+Optim.batchSize, lData.size());
            size_t nbData = end-start;
            
            std::vector<double> dC(outputSize*nbData);
            
            fwdProp  (lData.begin()+start, lData.begin()+end);
            calcDCost(lData.begin()+start, lData.begin()+end, dC);
            bwdProp  (dC);
            
            updateParams();
        }
        
        double timeEpoch = ( std::clock() - startTimeEpoch ) / (double) CLOCKS_PER_SEC;
        debugFile << "time " << timeEpoch << "s,";
        std::cout << "time " << timeEpoch << "s,";
        
        test(data.getTrainLabelData());
        double trainErrRate = errRate;
        double trainCost    = cost;
        
        test(data.getCrossLabelData());
        double crossErrRate = errRate;
        double crossCost    = cost;
        
        test(data.getTestLabelData());
        double testErrRate = errRate;
        double testCost    = cost;
        
        debugFile << trainErrRate << "," << crossErrRate << "," << testErrRate << ",";
        debugFile << trainCost    << "," << crossCost    << "," << testCost    << std::endl;
        std::cout << trainErrRate << "," << crossErrRate << "," << testErrRate << ",";
        std::cout << trainCost    << "," << crossCost    << "," << testCost    << std::endl;
    }
    
    debugFile.close();
}
//
void NeuralNetwork::test(const std::vector<LabelData>& lData)
{
    cost   =0.;
    errRate=0.;
    
    size_t totalISize = lData.size();
    size_t nbBatches  = (totalISize-1)/Optim.batchSize + 1;
    
    for (size_t batch=0; batch<nbBatches; ++batch)
    {
        size_t start  = batch*Optim.batchSize;
        size_t end    = std::min(start+Optim.batchSize, lData.size());
        size_t nbData = end-start;
        
        std::vector<double> dC(outputSize*nbData);
        
        fwdProp(lData.begin()+start, lData.begin()+end);
        cost += calcCost(lData.begin()+start, lData.begin()+end);
        errRate += isCorrect(lData.begin()+start, lData.begin()+end);
    }
    
    cost    /= lData.size();
    errRate  = 1.-errRate/lData.size();
}