//
//  CostFunc.h
//  NeuralNetworks
//
//  Created by Thomas Hubert on 08/04/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//

#ifndef NeuralNetworks_CostFunc_h
#define NeuralNetworks_CostFunc_h

#include "includes.h"

class CostFunc
{
public:
    CostFunc() {}
    virtual double  f(double a, double y) const = 0;
    virtual double df(double a, double y) const = 0;
};
//
class MSE : public CostFunc
{
public:
    MSE() {}
    double  f(double a, double y) const {return 0.5*(a-y)*(a-y);}
    double df(double a, double y) const {return (a-y);}
};
//
class CE : public CostFunc
{
public:
    CE() {}
    double  f(double a, double y) const {return -y*std::log(a) - (1-y)*std::log(1-a);}
    double df(double a, double y) const {return -(y-a)/(a*(1-a));}
};

#endif
