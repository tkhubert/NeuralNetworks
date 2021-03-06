//
//  CostFunc.h
//  NeuralNetworks
//
//  Created by Thomas Hubert on 08/04/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//

#ifndef NeuralNetworks_CostFunc_h
#define NeuralNetworks_CostFunc_h

#include "NN.h"
#include "Data.h"

namespace NN {
    
class CostFunc
{
public:
    CostFunc() {}
    virtual ~CostFunc() {}
    
    virtual string getName() const noexcept = 0;
    virtual real   f(const vec_r& a, LabelDataCItr dataStart, LabelDataCItr dataEnd) const noexcept = 0;
    virtual vec_r df(const vec_r& a, LabelDataCItr dataStart, LabelDataCItr dataEnd) const noexcept = 0;
};
//
    
//
class MSECostFunc : public CostFunc
{
public:
    MSECostFunc() {}
    
    string getName() const noexcept override {return "MSECFunc";}
    //
    real f(const vec_r& a, LabelDataCItr dataStart, LabelDataCItr dataEnd) const noexcept override
    {
        real val        = 0;
        auto nbData     = distance(dataStart, dataEnd);
        auto outputSize = a.size()/nbData;

        for (size_t d=0; d<nbData; ++d)
        {
            auto s = a.cbegin() + d*outputSize;
            auto y = (dataStart+d)->label;
            
            for (size_t i=0; i<outputSize; ++i)
            {
                auto label = i==y;
                auto aVal  = *(s+i);
                val+= 0.5*(aVal-label)*(aVal-label);
            }
        }
        return val;
    }
    //
    vec_r df(const vec_r& a, LabelDataCItr dataStart, LabelDataCItr dataEnd) const noexcept override
    {
        auto nbData     = distance(dataStart, dataEnd);
        auto outputSize = a.size()/nbData;
        
        vec_r dc(a.size());
        
        for (size_t d=0; d<nbData; ++d)
        {
            auto s = a.cbegin() + d*outputSize;
            auto y = (dataStart+d)->label;
            
            for (size_t i=0; i<outputSize; ++i)
            {
                auto label = i==y;
                auto aVal  = *(s+i);
                dc[d*outputSize+i] = aVal-label;
            }
        }
        return dc;
    }

};
//
    
//
class CECostFunc : public CostFunc
{
public:
    CECostFunc() {}
    
    string getName() const noexcept override {return "CECFunc";}
    //
    real f(const vec_r& a, LabelDataCItr dataStart, LabelDataCItr dataEnd) const noexcept override
    {
        real val        = 0;
        auto nbData     = distance(dataStart, dataEnd);
        auto outputSize = a.size()/nbData;
        
        for (size_t d=0; d<nbData; ++d)
        {
            auto s = a.cbegin() + d*outputSize;
            auto y = (dataStart+d)->label;
            
            for (size_t i=0; i<outputSize; ++i)
            {
                auto label = i==y;
                auto aVal  = *(s+i);
                val+= -label*log(aVal) - (1-label)*log(1-aVal);
            }
        }
        return val;
    }
    //
    vec_r df(const vec_r& a, LabelDataCItr dataStart, LabelDataCItr dataEnd) const noexcept override
    {
        auto nbData     = distance(dataStart, dataEnd);
        auto outputSize = a.size()/nbData;
        
        vec_r dc(a.size());
        
        for (size_t d=0; d<nbData; ++d)
        {
            auto s = a.cbegin() + d*outputSize;
            auto y = (dataStart+d)->label;
            
            auto aLabel = *(s+y);
            for (size_t i=0; i<outputSize; ++i)
            {
                auto aVal  = *(s+i);
                dc[d*outputSize+i] = 1/(1-aVal);
            }
            dc[d*outputSize+y] = -1/aLabel;
        }
        return dc;
    }
};
//
    
//
class SMCostFunc : public CostFunc
{
public:
    SMCostFunc() {}
    
    string getName() const noexcept override {return "SMCFunc";}
    //
    real f(const vec_r& a, LabelDataCItr dataStart, LabelDataCItr dataEnd) const noexcept override
    {
        real val        = 0;
        auto nbData     = distance(dataStart, dataEnd);
        auto outputSize = a.size()/nbData;

        for (size_t d=0; d<nbData; ++d)
        {
            auto s    = a.cbegin() + d*outputSize;
            auto e    = s + outputSize;
            auto y    = (dataStart+d)->label;
            auto maxA = *(max_element(s, e));
            
            vec_r expA(outputSize);
            transform(s, e, expA.begin(), [maxA](auto x) {return exp(x-maxA);});
            auto sum = accumulate(expA.begin(), expA.end(), 0.);
            
            val -= log(expA[y]/sum);
        }
        return val;
    }
    //
    vec_r df(const vec_r& a, LabelDataCItr dataStart, LabelDataCItr dataEnd) const noexcept override
    {
        auto nbData     = distance(dataStart, dataEnd);
        auto outputSize = a.size()/nbData;
        
        vec_r dc(a.size());

        for (size_t d=0; d<nbData; ++d)
        {
            auto s    = a.cbegin() + d*outputSize;
            auto e    = s + outputSize;
            auto y    = (dataStart+d)->label;
            auto maxA = *(max_element(s, e));
            
            vec_r expA(outputSize);
            transform(s, e, expA.begin(), [maxA](auto x) {return exp(x-maxA);});
            auto sum = accumulate(expA.begin(), expA.end(), 0.);
            
            for (size_t o=0; o<outputSize; ++o)
                dc[d*outputSize+o] = expA[o]/sum;
            
            dc[d*outputSize+y] -=1;
        }
        return dc;
    }
};
//
    
//
class SVMCostFunc : public CostFunc
{
public:
    SVMCostFunc() {}
    
    string getName() const noexcept override {return "SVMCFunc";}
    //
    real f(const vec_r& a, LabelDataCItr dataStart, LabelDataCItr dataEnd) const noexcept override
    {
        real val        = 0;
        auto nbData     = distance(dataStart, dataEnd);
        auto outputSize = a.size()/nbData;
        
        for (size_t d=0; d<nbData; ++d)
        {
            auto s = a.cbegin() + d*outputSize;
            auto y = (dataStart+d)->label;
            
            auto aLabel = *(s+y);
            for (size_t i=0; i<outputSize; ++i)
            {
                auto aVal  = *(s+i);
                val+= max(0.,aVal-aLabel+1.);
            }
            val -=1;
        }
        return val;
    }
    //
    vec_r df(const vec_r& a, LabelDataCItr dataStart, LabelDataCItr dataEnd) const noexcept override
    {
        auto nbData     = distance(dataStart, dataEnd);
        auto outputSize = a.size()/nbData;
        
        vec_r dc(a.size());
        
        for (size_t d=0; d<nbData; ++d)
        {
            auto s = a.cbegin() + d*outputSize;
            auto y = (dataStart+d)->label;
            
            auto aLabel = *(s+y);
            for (size_t i=0; i<outputSize; ++i)
            {
                auto aVal  = *(s+i);
                dc[d*outputSize+i] = aVal-aLabel>-1 ? 1. : 0.;
            }
            for (size_t i=0; i<outputSize; ++i)
            {
                auto aVal  = *(s+i);
                dc[d*outputSize+y] += aVal-aLabel>-1 ? -1. : 0.;
            }
        }
        return dc;
    }
};
//

}

#endif
