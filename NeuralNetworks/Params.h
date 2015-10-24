//
//  Params.h
//  NeuralNetworks
//
//  Created by Thomas Hubert on 24/10/2015.
//
//

#ifndef Params_h
#define Params_h

namespace NN
{
class Params
{
public:
    Params() : nbData(0), nbBias(0), nbWeight(0), weightInputSize(1) { params.resize(0); }
    //
    Params(size_t nbBias, size_t nbWeight, size_t weightInputSize) :
    nbData(nbBias+nbWeight),
    nbBias(nbBias),
    nbWeight(nbWeight),
    weightInputSize(weightInputSize)
    {
        params.resize(nbData);
    }
    //
    void resize(size_t _nbBias, size_t _nbWeight, size_t _weightInputSize)
    {
        nbData          = _nbBias+_nbWeight;
        nbBias          = _nbBias;
        nbWeight        = _nbWeight;
        weightInputSize = _weightInputSize;
        params.resize(nbData);
    }
    //
    void initParams(default_random_engine& gen)
    {
        normal_distribution<real> norm(0.,1.);
        real normalizer = 1./sqrt(weightInputSize);
        
        size_t o=0;
        for (; o<nbBias; ++o)
            params[o] = norm(gen);
        for (; o<params.size(); ++o)
            params[o] = norm(gen)*normalizer;
    }
    //
    const auto&       get()         const {return params;}
    auto&             get()               {return params;}
    auto              getNbBias()   const {return nbBias;}
    auto              getNbWeight() const {return nbWeight;}
    auto              size()        const {return nbData;}
    const auto* const getCBPtr()    const {return &params[0];}
    const auto* const getCWPtr()    const {return &params[nbBias];}
    auto*             getBPtr()           {return &params[0];}
    auto*             getWPtr()           {return &params[nbBias];}
    //
    void reset() { fill(params.begin(), params.end(), 0.); }
    //
    
private:
    vec_r  params;
    size_t nbData;
    size_t nbBias;
    size_t nbWeight;
    size_t weightInputSize;
};
}
#endif /* Params_h */
