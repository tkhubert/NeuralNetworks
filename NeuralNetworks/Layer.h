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

template<typename ActFunc>
class Layer
{
public:
    Layer(size_t _size);
    
    virtual std::string getName()    const = 0;
    virtual std::string getDetails() const = 0;
    
    const std::vector<double>& getA()         const {return a; }
    const std::vector<double>& getdA()        const {return da; }
    const std::vector<double>& getBias()      const {return bias; }
    const std::vector<double>& getWeight()    const {return weight; }
    const Layer*               getNextLayer() const {return nextLayer;}
    const Layer*               getPrevLayer() const {return prevLayer;}
    
    virtual void fwdProp() = 0;
    virtual void bwdProp() = 0;

    void setNextLayer(Layer* next)    { nextLayer=next; }
    void setPrevLayer(Layer* prev)    { prevLayer=prev; }
    void setDelta    (const std::vector<double>& _delta) {delta = _delta;}
    
protected:
    size_t              size;
    
    std::vector<double>  a;
    std::vector<double> da;
    std::vector<double> delta;
    
    std::vector<double> bias;
    std::vector<double> weight;
    
    Layer<ActFunc>* nextLayer;
    Layer<ActFunc>* prevLayer;
    
    ActFunc AFunc;
};
//
template<typename ActFunc>
class InputLayer : public Layer<ActFunc>
{
public:
    InputLayer(size_t _size);
    std::string getName()    const {return "InputLayer";}
    std::string getDetails() const {return "";}
    
    void fwdProp() {throw "InputLayer do not inplement fwdProp()";}
    void bwdProp() {throw "InputLayer do not inplement bwdProp()";}
    
    void setInput(const std::vector<double> input) { this->a = input;}
};
//
template<typename ActFunc>
class FCLayer : public Layer<ActFunc>
{
public:
    FCLayer(size_t _size);
    std::string getName()    const {return "FCLayer";}
    std::string getDetails() const {return "";}
    
    void fwdProp();
    void bwdProp();
};



#endif /* defined(__NeuralNetworks__Layer__) */
