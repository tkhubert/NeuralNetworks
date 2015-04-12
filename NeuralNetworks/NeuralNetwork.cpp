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
    
    debugFile  = std::ofstream("/Users/tkhubert/Documents/Projects/NeuralNetworks/MNist/debugNN.csv");
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
        layers[i]->updateParams(Optim.alpha);
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
    const std::vector<std::vector<double> >& inputs = data.getTrainData();
    const std::vector<int>&                  labels = data.getTrainLabels();
    
    debugFile << "Start training -------------------------------------"<<std::endl;
    
    size_t nbBatches  = (inputs.size()-1)/Optim.batchSize + 1;
    
    for (size_t t=0; t<Optim.nbEpochs; ++t)
    {
        debugFile << "Epoch: " << t << ", ";
        std::cout << "Epoch: " << t << ", ";
        std::clock_t startTimeEpoch = std::clock();;
        
        for (size_t batch=0; batch<nbBatches; ++batch)
        {
            size_t start = batch*Optim.batchSize;
            size_t end   = std::min(start+Optim.batchSize, inputs.size());
            
            double batchCost = 0.;
            std::vector<double> dc(outputSize);
            for (size_t i=start; i<end; ++i)
            {
                fwdProp(inputs[i]);
                
                batchCost += calcCost(labels[i])/(end-start);
                calcDCost(labels[i], dc);
                for (size_t j=0; j<outputSize; ++j)
                    dc[j] /=(end-start);
                
                bwdProp(dc);
            }
            
//            const double tweakSize = 0.0001;
//            for (size_t k=1; k<nbLayers; ++k)
//            {
//                Layer*              l = layers[k];
//                std::vector<double> w = l->getWeight();
//                
//                for (int j=0; j<2; ++j)
//                {
//                    int    idx = rand() % w.size();
//                    double tmp = w[idx];
//                    w[idx] +=tweakSize;
//                    l->setWeight(w);
//                    
//                    double batchCost2 = 0.;
//                    for (size_t i=start; i<end; ++i)
//                    {
//                        setInput(inputs[i]);
//                        fwdProp();
//                        
//                        batchCost2 += calcCost(labels[i]);
//                    }
//                    batchCost2 /= (end-start);
//                    w[idx] = tmp-tweakSize;
//                    l->setWeight(w);
//                    
//                    double batchCost3 = 0.;
//                    for (size_t i=start; i<end; ++i)
//                    {
//                        setInput(inputs[i]);
//                        fwdProp();
//                        
//                        batchCost3 += calcCost(labels[i]);
//                    }
//                    batchCost3 /= (end-start);
//                    w[idx] = tmp;
//                    l->setWeight(w);
//                    
//                    double grad  = (batchCost2 - batchCost3)/(2*tweakSize);
//                    double grad2 = l->getdWeight()[idx];
//                    double error = grad2==0 ? 0 : grad==0 ? 0 : 1-grad2/grad;
//                    if (abs(error)>0.01)
//                        std:: cout << "PROBLEM" << k << end;;
//                }
//            }
            
            
            updateParams();
        }
        
        double timeEpoch = ( std::clock() - startTimeEpoch ) / (double) CLOCKS_PER_SEC;
        debugFile << "time " << timeEpoch << "s,";
        //std::cout << "time " << timeEpoch << "s,";
        
        test(inputs, labels);
        double trainErrRate = errRate;
        double trainCost    = cost;
        
        test(data.getCrossData(), data.getCrossLabels());
        debugFile << trainErrRate << "," << errRate << ",";
        debugFile << trainCost    << "," << cost << std::endl;
        std::cout << trainErrRate << "," << errRate << ", ";
        std::cout << trainCost    << "," << cost << std::endl;
    }
    
    test(data.getTestData(), data.getTestLabels());
    debugFile << "Test," << " , " << errRate << ", , " << cost << std::endl;
    std::cout << "Test," << " , " << errRate << ", , " << cost << std::endl;
    debugFile.close();
}
//
void NeuralNetwork::test(const std::vector<std::vector<double> >& inputs, const std::vector<int>& labels)
{
    cost=0.;
    errRate=0.;
    
    for (size_t i=0; i<inputs.size(); ++i)
    {
        fwdProp(inputs[i]);
        cost    += calcCost(labels[i]);
        errRate += isCorrect(labels[i]);
    }
    
    cost    /= inputs.size();
    errRate  = 1.-errRate/inputs.size();
}