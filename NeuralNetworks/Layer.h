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
    Layer(size_t size, float dropRate, const ActivationFunc& AFunc);
    ~Layer();
    
    virtual string     getName()    const = 0;
    virtual string     getDetails() const = 0;
    virtual LayerClass getClass()   const = 0;
    
    auto        getInputSize()   const {return inputSize;}
    auto        getOutputSize()  const {return outputSize;}
    const auto& getA()           const {return a; }
    const auto& getDrop()        const {return drop; }
    const auto& getBias()        const {return bias; }
    const auto& getWeight()      const {return weight; }
    auto&       getDelta()             {return delta; }

    const Layer*          getNextLayer() const {return nextLayer;}
    const Layer*          getPrevLayer() const {return prevLayer;}
    const ActivationFunc& getAFunc()     const {return AFunc;}

    void setNbData   (size_t _nbData)          { resize(_nbData);}
    void setNextLayer(Layer* next)             { nextLayer = next; }
    virtual void setPrevLayer(Layer* prev) = 0;
    void setPhase    (Phase  p)                { phase = p; }
    void setA        (const vector<float>& _a) { a = _a;}
    void setA        (vector<float>&&      _a) { a = move(_a);}
    void setDrop     ();
    
    virtual void setDCost(const vector<float>& dc);
    virtual void fwdProp()  = 0;
    virtual void bwdProp()  = 0;
    virtual void calcGrad() = 0;
    void         initParams();
    void         updateParams(float alpha, float friction, float lambdaOverN);
    
protected:
    static size_t layerCount;
    size_t        layerNb;
    size_t        inputSize;
    size_t        outputSize;
    size_t        nbData;
    float         dropRate;
    Phase         phase;
    
    vector<float> a;
    vector<float> delta;
    vector<float> drop;
    
    vector<float> bias;
    vector<float> dbias;
    vector<float> vbias;
    vector<float> weight;
    vector<float> dweight;
    vector<float> vweight;
    // w[o][i] = w[o*InputSize + i] or = w[i*OutputSize+o]

    Layer* nextLayer;
    Layer* prevLayer;

    const ActivationFunc& AFunc;
    default_random_engine gen;
    
    void resize(size_t _nbData);
};

}

#endif /* defined(__NeuralNetworks__Layer__) */
