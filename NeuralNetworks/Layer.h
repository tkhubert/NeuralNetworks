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
class FCLayer : public Layer<ActFunc>
{
public:
    FCLayer(size_t _size);
    
    void fwdProp();
    void bwdProp();
};



#endif /* defined(__NeuralNetworks__Layer__) */
