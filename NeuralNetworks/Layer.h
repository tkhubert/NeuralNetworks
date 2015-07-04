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
    
    size_t                     getInputSize()   const {return inputSize;}
    size_t                     getOutputSize()  const {return outputSize;}
    const std::vector<double>& getA()           const {return a; }
    const std::vector<double>& getBias()        const {return bias; }
    const std::vector<double>& getWeight()      const {return weight; }
    
    const std::vector<double>& getdA()          const {return da; }
    std::vector<double>&       getDelta()             {return delta; }
    std::vector<double>&       getdBias()             {return dbias; }
    std::vector<double>&       getdWeight()           {return dweight; }

    const Layer*               getNextLayer()   const {return nextLayer;}
    const Layer*               getPrevLayer()   const {return prevLayer;}
    const ActivationFunc&      getAFunc()       const {return AFunc;}

    void setNextLayer(Layer* next)    { nextLayer=next; }
    void setPrevLayer(Layer* prev)    { prevLayer=prev; }
    void setA        (const std::vector<double>& _a)      {a      = _a;}
    void setDelta    (const std::vector<double>& _delta)  {delta  = _delta;}
    void setWeight   (const std::vector<double>& _weight) {weight = _weight;}
    void setBias     (const std::vector<double>& _bias)   {bias   = _bias;}
    
    virtual void setDCost(const std::vector<double>& dc);
    virtual void fwdProp()  = 0;
    virtual void bwdProp()  = 0;
    virtual void calcGrad() = 0;
    void         initParams();
    void         updateParams(double alpha, double lambdaOverN);
    
protected:
    size_t              inputSize;
    size_t              outputSize;
    
    std::vector<double> a;
    std::vector<double> bias;
    std::vector<double> weight;
    // w[o][i] = w[o*InputSize + i] or = w[i*OutputSize+o]
    
    double weightSqSum;
    
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
    FCLayer(size_t _inputSize, size_t _outputSize, const ActivationFunc& _AFunc) : Layer(_inputSize, _outputSize, _AFunc) {};
    std::string getName()    const {return "FCLayer";}
    std::string getDetails() const {return "";}
    
    void fwdProp();
    void bwdProp();
    void calcGrad();
};



#endif /* defined(__NeuralNetworks__Layer__) */
