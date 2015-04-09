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
    virtual double  f(const std::vector<double>& a, const std::vector<double>& y) const = 0;
    virtual void df(const std::vector<double>& a, const std::vector<double>& y, std::vector<double>& da) const = 0;
};
//
class MSE : public CostFunc
{
public:
    MSE() {}
    double  f(const std::vector<double>& a, const std::vector<double>& y) const
    {
        double val=0.;
        for (size_t i=0; i<a.size(); ++i)
            val+= (a[i]-y[i])*(a[i]-y[i]);
        return 0.5*val;
    }
    //
    void df(const std::vector<double>& a, const std::vector<double>& y, std::vector<double>& da) const
    {
        for (size_t i=0; i<a.size(); ++i)
            da[i] = a[i]-y[i];
        return;
    }
};
//
class CE : public CostFunc
{
public:
    CE() {}
    double  f(const std::vector<double>& a, const std::vector<double>& y) const
    {
        double val=0.;
        for (size_t i=0; i<a.size(); ++i)
            val += -y[i]*std::log(a[i]) - (1-y[i])*std::log(1-a[i]);
        return val;
    }
    //
    void df(const std::vector<double>& a, const std::vector<double>& y, std::vector<double>& da) const
    {
        for (size_t i=0; i<a.size(); ++i)
            da[i] = -(y[i]-a[i])/(a[i]*(1-a[i]));
        return;
    }
};

#endif
