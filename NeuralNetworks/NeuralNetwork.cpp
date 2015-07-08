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
void NeuralNetwork::fwdProp(const std::vector<double>& input)
{
    setInput(input);
    for (size_t i=1; i<nbLayers; ++i)
        layers[i]->fwdProp();
}
//
void NeuralNetwork::bwdProp(const std::vector<double>& dc)
{
    setDCost(dc);
    for (size_t i=nbLayers-1; i>=1; --i)
        layers[i]->bwdProp();
}
//
const std::vector<double>& NeuralNetwork::predict(const std::vector<double>& input)
{
    fwdProp(input);
    return getOutput();
}
//
bool NeuralNetwork::isCorrect(int label) const
{
    const std::vector<double>& prediction = getOutput();
    return std::distance(prediction.begin(), std::max_element(prediction.begin(), prediction.end()))==label;
}
//
void NeuralNetwork::train(const DataContainer& data)
{
    std::vector<LabelData> ldata = data.getTrainLabelData();
    
    std::cout << "Start training " << getName() << "-------------"<<std::endl;
    
    size_t totalISize = ldata.size();
    size_t nbBatches  = (totalISize-1)/Optim.batchSize + 1;
    std::vector<double> dc(outputSize);
    
    for (size_t t=0; t<Optim.nbEpochs; ++t)
    {
        debugFile << t << ", ";
        std::cout << "Epoch: " << t << ", ";
        std::clock_t startTimeEpoch = std::clock();
        
        std::random_shuffle(ldata.begin(), ldata.end());
        
        for (size_t batch=0; batch<nbBatches; ++batch)
        {
            size_t start = batch*Optim.batchSize;
            size_t end   = std::min(start+Optim.batchSize, ldata.size());
            
            for (size_t i=start; i<end; ++i)
            {
                fwdProp(ldata[i].data);
                calcDCost(ldata[i].label, dc);
                bwdProp(dc);
            }
            
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
    
    for (size_t i=0; i<lData.size(); ++i)
    {
        fwdProp(lData[i].data);
        cost    += calcCost(lData[i].label);
        errRate += isCorrect(lData[i].label);
    }
    
    cost    /= lData.size();
    errRate  = 1.-errRate/lData.size();
}