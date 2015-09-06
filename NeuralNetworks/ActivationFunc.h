//
//  ActivationFunc.h
//  NeuralNetworks
//
//  Created by Thomas Hubert on 08/04/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//

#ifndef NeuralNetworks_ActivationFunc_h
#define NeuralNetworks_ActivationFunc_h

#include "NN.h"

namespace NN {
    
class ActivationFunc
{
public:
    ActivationFunc() {}
    virtual string getName()   const = 0;
    virtual float  f (float x) const = 0;
    virtual float  df(float f) const = 0;
};
//
class IdFunc: public ActivationFunc
{
public:
    IdFunc() {}
    
    string getName()  const {return "IdFunc";}
    float f (float x) const {return x;}
    float df(float f) const {return 1;}
};
//
class SigmoidFunc : public ActivationFunc
{
public:
    SigmoidFunc() {}
    
    string getName()  const {return "SigAFunc";}
    float f (float x) const {return 1./(1+exp(-x));}
    float df(float f) const {return f*(1-f);}
};
//
class TanHFunc : public ActivationFunc
{
public:
    TanHFunc() {}
    
    string getName() const {return "TanHAFunc";}
    float f (float x) const
    {
        float tmp1 = exp(x);
        float tmp2 = 1/tmp1;
        return (tmp1-tmp2)/(tmp1+tmp2);
    }
    float df(float f) const {return 1.-f*f;}
};
//
class RLFunc : public ActivationFunc
{
public:
    string getName() const {return "RLAFunc";}
    RLFunc(float _a=1., float _b=0.001) : a(_a), b(_b) {}
    float f (float x) const {return x>=0.f ? a*x : b*x;}
    float df(float f) const {return f>=0.f ? a   : b;}
    
private:
    float a;
    float b;
};
    
}
#endif
