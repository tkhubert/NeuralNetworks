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
#include "Data.h"

class CostFunc
{
public:
    CostFunc() {}
    virtual std::string getName() const = 0;
    virtual float      f(const std::vector<float>& a, int y) const = 0;
    virtual void        df(const std::vector<float>& a, int y, std::vector<float>& dc) const = 0;
    //
    virtual float f(const std::vector<float>& a, std::vector<LabelData>::const_iterator dataStart, std::vector<LabelData>::const_iterator dataEnd) const
    {
        float tmp        = 0;
        size_t nbData     = std::distance(dataStart, dataEnd);
        size_t outputSize = a.size()/nbData;
        
        std::vector<float> aloc(outputSize);
        
        for (size_t d=0; d<nbData; ++d)
        {
            std::vector<float>::const_iterator s = a.begin() + d*outputSize;
            std::vector<float>::const_iterator e = s + outputSize;
            std::copy(s, e, aloc.begin());
            
            const LabelData& lD = *(dataStart+d);
            tmp += f(aloc, lD.label);
        }
        return tmp;
    }
    //
    virtual void df(const std::vector<float>& a, std::vector<LabelData>::const_iterator dataStart, std::vector<LabelData>::const_iterator dataEnd, std::vector<float>& dc) const
    {
        size_t nbData     = std::distance(dataStart, dataEnd);
        size_t outputSize = a.size()/nbData;
        
        std::vector<float> aloc(outputSize);
        std::vector<float> dcloc(outputSize);
        
        for (size_t d=0; d<nbData; ++d)
        {
            std::vector<float>::const_iterator s = a.begin() + d*outputSize;
            std::vector<float>::const_iterator e = s + outputSize;
            std::copy(s, e, aloc.begin());
            
            const LabelData& lD = *(dataStart+d);
            df(aloc, lD.label, dcloc);
            
            for (size_t o=0; o<outputSize; ++o)
                dc[d*outputSize+o] = dcloc[o];
        }
    }
};
//
class MSECostFunc : public CostFunc
{
public:
    MSECostFunc() {}
    
    std::string getName() const {return "MSECFunc";}
    
    float  f(const std::vector<float>& a, int y) const
    {
        float val=0.;
        for (size_t i=0; i<a.size(); ++i)
        {
            int label = i==y;
            val+= (a[i]-label)*(a[i]-label);
        }
        return 0.5*val;
    }
    //
    void df(const std::vector<float>& a, int y, std::vector<float>& dc) const
    {
        for (size_t i=0; i<a.size(); ++i)
        {
            int label = i==y;
            dc[i] = a[i]-label;
        }
    }
};
//
class CECostFunc : public CostFunc
{
public:
    CECostFunc() {}
    
    std::string getName() const {return "CECFunc";}
    
    float  f(const std::vector<float>& a, int y) const
    {
        float val=0.;
        for (size_t i=0; i<a.size(); ++i)
        {
            int label = i==y;
            val += -label*std::log(a[i]) - (1-label)*std::log(1-a[i]);
        }
        return val;
    }
    //
    void df(const std::vector<float>& a, int y, std::vector<float>& dc) const
    {
        for (size_t i=0; i<a.size(); ++i)
            dc[i] = 1/(1-a[i]);
        
        dc[y] = -1/a[y];
    }
};
//
class SMCostFunc : public CostFunc
{
public:
    SMCostFunc() {}
    
    std::string getName() const {return "SMCFunc";}
    //
    struct SExp : std::unary_function<float, float>
    {
        float m;
        
        SExp(float m) : m(m) {}
        
        float operator()(float x) const {return exp(x-m);}
    };
    //
    float f(const std::vector<float>& a, int y) const
    {
        float maxA = *(std::max_element(a.begin(), a.end()));
        std::vector<float> expA(a.size());
        
        float sum = 0.;
        for (size_t i=0; i<a.size(); ++i)
        {
            expA[i] = exp(a[i]-maxA);
            sum += expA[i];
        }
        
        return -log(expA[y]/sum);
    }
    //
    void df(const std::vector<float>&a, int y, std::vector<float>& dc) const
    {
        std::vector<float> expA(a.size());
        
        float sum = 0.;
        for (size_t i=0; i<a.size(); ++i)
        {
            expA[i] = exp(a[i]);
            sum += expA[i];
        }
        
        for (size_t i=0; i<a.size(); ++i)
            dc[i] = expA[i]/sum;
        
        dc[y] -=1;
    }
    //
    float f(const std::vector<float>& a, std::vector<LabelData>::const_iterator dataStart, std::vector<LabelData>::const_iterator dataEnd) const
    {
        float val        = 0;
        size_t nbData     = std::distance(dataStart, dataEnd);
        size_t outputSize = a.size()/nbData;

        for (size_t d=0; d<nbData; ++d)
        {
            std::vector<float>::const_iterator s = a.begin() + d*outputSize;
            std::vector<float>::const_iterator e = s + outputSize;
            
            int y = (dataStart+d)->label;
            
            float maxA = *(std::max_element(s, e));
            std::vector<float> expA(outputSize);
            
            std::transform(s, e, expA.begin(), SExp(maxA));
            float sum = std::accumulate(expA.begin(), expA.end(), 0.);

            val -= log(expA[y]/sum);
        }
        return val;
    }
    //
    void df(const std::vector<float>& a, std::vector<LabelData>::const_iterator dataStart, std::vector<LabelData>::const_iterator dataEnd, std::vector<float>& dc) const
    {
        size_t nbData     = std::distance(dataStart, dataEnd);
        size_t outputSize = a.size()/nbData;

        for (size_t d=0; d<nbData; ++d)
        {
            std::vector<float>::const_iterator s = a.begin() + d*outputSize;
            std::vector<float>::const_iterator e = s + outputSize;
            
            int y = (dataStart+d)->label;
            std::vector<float> expA(outputSize);
            
            float maxA = *(std::max_element(s, e));
            
            std::transform(s, e, expA.begin(), SExp(maxA));
            float sum = std::accumulate(expA.begin(), expA.end(), 0.);
            
            for (size_t o=0; o<outputSize; ++o)
                dc[d*outputSize+o] = expA[o]/sum;
            
            dc[d*outputSize+y] -=1;
        }
    }
};
//
class SVMCostFunc : public CostFunc
{
public:
    SVMCostFunc() {}
    
    std::string getName() const {return "SVMCFunc";}
    
    float f(const std::vector<float>& a, int y) const
    {
        float val = -1.;
        for (size_t i=0; i<a.size(); ++i)
            val += std::max(0.f,a[i]-a[y]+1.f);
        return val;
    }
    //
    void df(const std::vector<float>&a, int y, std::vector<float>& dc) const
    {
        for (size_t i=0; i<a.size(); ++i)
            dc[i] = a[i]-a[y]>-1 ? 1. : 0.;

        for (size_t i=0; i<a.size(); ++i)
            dc[y] += a[i]-a[y]>-1 ? -1. : 0;
    }
};
//

#endif
