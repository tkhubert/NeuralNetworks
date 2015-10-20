//
//  ConvLayer.h
//  NeuralNetworks
//
//  Created by Thomas Hubert on 15/08/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//

#ifndef __NeuralNetworks__ConvLayer__
#define __NeuralNetworks__ConvLayer__

#include "Layer.h"

namespace NN
{
    
class ConvLayer : public Layer
{
public:
    ConvLayer(size_t width, size_t height, size_t depth, size_t mapSize, size_t stride, const ActivationFunc& AFunc);
    
    string getName()      const {return "ConvLayer";}
    string getDetails()   const {return "";}
    LayerClass getClass() const {return LayerClass::ConvLayer;}
    
    auto getWidth()   const {return width;}
    auto getHeight()  const {return height;}
    auto getDepth()   const {return depth;}
    auto getMapSize() const {return mapSize;}
    auto getStride()  const {return stride;}
    
    virtual void setPrevLayer(Layer* layer);
    virtual void fwdProp();
    virtual void bwdProp();
    virtual void calcGrad();
    
protected:
    size_t width;
    size_t height;
    size_t depth;
    size_t mapSize;
    size_t stride;
    
private:
    void naiveFwdProp();
    void naiveBwdProp();
    void naiveCalcGrad();
    void img2MatFwdProp();
    void img2MatBwdProp();
    void img2MatCalcGrad();
    
    void genPrevAMatFwd(size_t d, vec_r& prevAMat) const;
    void genPrevAMatGrad(size_t d, vec_r& prevAMat) const;
    void genDeltaMat(size_t d, vector<int>& hIdxVec, vector<int>& wIdxVec, vec_r& deltaMat) const;
    void genWeightMat(vec_r& weightMat) const;
    void genIdxVec(size_t pdim, size_t dim, vector<int>& weightIdxVec) const;

};

}

#endif /* defined(__NeuralNetworks__ConvLayer__) */
