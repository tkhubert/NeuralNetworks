//
//  Layer.h
//  NeuralNetworks
//
//  Created by Thomas Hubert on 08/04/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//

#ifndef __NeuralNetworks__Layer__
#define __NeuralNetworks__Layer__

#include "includes.h"
#include "ActivationFunc.h"

class Layer
{
public:
    Layer(size_t _inputSize, size_t _outputSize, const ActivationFunc& _AFunc);
    ~Layer();
    
    virtual std::string getName()    const = 0;
    virtual std::string getDetails() const = 0;
    
    auto        getInputSize()   const {return inputSize;}
    auto        getOutputSize()  const {return outputSize;}
    const auto& getA()           const {return a; }
    const auto& getBias()        const {return bias; }
    const auto& getWeight()      const {return weight; }
    auto&       getDelta()             {return delta; }

    const Layer*          getNextLayer() const {return nextLayer;}
    const Layer*          getPrevLayer() const {return prevLayer;}
    const ActivationFunc& getAFunc()     const {return AFunc;}

    void setNbData   (size_t _nbData)                    { resize(_nbData);}
    void setNextLayer(Layer* next)                       { nextLayer = next; }
    void setPrevLayer(Layer* prev)                       { prevLayer = prev; }
    void setA        (const std::vector<float>& _a)      { a         = _a;}
    void setDelta    (const std::vector<float>& _delta)  { delta     = _delta;}
    void setWeight   (const std::vector<float>& _weight) { weight    = _weight;}
    void setBias     (const std::vector<float>& _bias)   { bias      = _bias;}
    
    virtual void setDCost(const std::vector<float>& dc);
    virtual void fwdProp()  = 0;
    virtual void bwdProp()  = 0;
    virtual void calcGrad() = 0;
    void         initParams();
    void         updateParams(float alpha, float friction, float lambdaOverN);
    
protected:
    static size_t      layerCount;
    size_t             layerNb;
    size_t             inputSize;
    size_t             outputSize;
    size_t             nbData;
    
    std::vector<float> a;
    std::vector<float> delta;
    
    std::vector<float> bias;
    std::vector<float> dbias;
    std::vector<float> vbias;
    std::vector<float> weight;
    std::vector<float> dweight;
    std::vector<float> vweight;
    // w[o][i] = w[o*InputSize + i] or = w[i*OutputSize+o]

    Layer* nextLayer;
    Layer* prevLayer;

    const ActivationFunc& AFunc;
    
    void resize(size_t _nbData);
};


#endif /* defined(__NeuralNetworks__Layer__) */
