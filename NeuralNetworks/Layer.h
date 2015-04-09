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
    
    size_t                     getSize()      const {return inputSize;}
    const std::vector<double>& getA()         const {return a; }
    const std::vector<double>& getdA()        const {return da; }
    const std::vector<double>& getDelta()     const {return delta; }
    const std::vector<double>& getBias()      const {return bias; }
    const std::vector<double>& getWeight()    const {return weight; }
    const Layer*               getNextLayer() const {return nextLayer;}
    const Layer*               getPrevLayer() const {return prevLayer;}

    void setNextLayer(Layer* next)    { nextLayer=next; }
    void setPrevLayer(Layer* prev)    { prevLayer=prev; }
    void setA        (const std::vector<double>& _a)     {a = _a;}
    void setDelta    (const std::vector<double>& _delta) {delta = _delta;}
    
    virtual void setDCost(const std::vector<double>& dc);
    virtual void fwdProp() = 0;
    virtual void bwdProp() = 0;
    
protected:
    size_t              inputSize;
    size_t              outputSize;
    
    std::vector<double> a;
    std::vector<double> bias;
    std::vector<double> weight;
    
    std::vector<double> da;
    std::vector<double> dbias;
    std::vector<double> dweight;
    std::vector<double> delta;
    Layer* nextLayer;
    Layer* prevLayer;

    const ActivationFunc& AFunc;
};
//
class FCLayer : public Layer
{
public:
    FCLayer(size_t _size);
    std::string getName()    const {return "FCLayer";}
    std::string getDetails() const {return "";}
    
    void fwdProp();
    void bwdProp();
};



#endif /* defined(__NeuralNetworks__Layer__) */
