//
//  Layer.h
//  NeuralNetworks
//
//  Created by Thomas Hubert on 08/04/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//

#ifndef __NeuralNetworks__Layer__
#define __NeuralNetworks__Layer__

#include "NN.h"
#include "ActivationFunc.h"

namespace NN {

enum class Phase { TRAIN, TEST};
enum class LayerClass { FCLayer, ConvLayer, ConvPoolLayer };
//
class Layer
{
public:
    Layer(size_t size, real dropRate, const ActivationFunc& AFunc);
    ~Layer();
    
    virtual string     getName()    const = 0;
    virtual string     getDetails() const = 0;
    virtual LayerClass getClass()   const = 0;
    
    auto        getInputSize () const {return inputSize;}
    auto        getOutputSize() const {return outputSize;}
    auto        getLayerNb   () const {return layerNb;}
    const auto& getA         () const {return a; }
    const auto& getDrop      () const {return drop; }
    const auto& getBias      () const {return bias; }
    auto&       getBias      ()       {return bias; }
    auto&       getDBias     ()       {return dbias;}
    const auto& getWeight    () const {return weight; }
    auto&       getWeight    ()       {return weight;}
    auto&       getDWeight   ()       {return dweight; }
    auto&       getDelta     ()       {return delta; }

    const Layer*          getNextLayer() const {return nextLayer;}
    const Layer*          getPrevLayer() const {return prevLayer;}
    const ActivationFunc& getAFunc()     const {return AFunc;}

    void setNbData   (size_t _nbData)          { resize(_nbData);}
    void setNextLayer(Layer* next)             { nextLayer = next; }
    virtual void setPrevLayer(Layer* prev) = 0;
    void setPhase    (Phase  p)                { phase = p; }
    void setA        (const vec_r& _a) { a = _a;}
    void setA        (vec_r&&      _a) { a = move(_a);}
    void setDrop     ();
    
    virtual void setDCost(const vec_r& dc);
    virtual void fwdProp()  = 0;
    virtual void bwdProp()  = 0;
    virtual void calcGrad() = 0;
    virtual void regularize(real lambda);

    void initParams();
    
protected:
    static size_t layerCount;
    size_t        layerNb;
    size_t        inputSize;
    size_t        outputSize;
    size_t        weightInputSize;
    size_t        nbData;
    real          dropRate;
    Phase         phase;
    
    vec_r a;
    vec_r delta;
    vec_r drop;
    
    vec_r bias;
    vec_r dbias;
    vec_r weight;
    vec_r dweight;
    // w[o][i] = w[o*InputSize + i] or = w[i*OutputSize+o]

    Layer* nextLayer;
    Layer* prevLayer;

    const ActivationFunc& AFunc;
    default_random_engine gen;
    
    virtual void resize(size_t nbData);
};

}

#endif /* defined(__NeuralNetworks__Layer__) */
