//
//  Trainer.h
//  NeuralNetworks
//
//  Created by Thomas Hubert on 11/05/2016.
//
//

#ifndef Trainer_h
#define Trainer_h

#include "NN.h"
#include "Optimizer.h"

namespace NN
{
    
class Trainer
{
public:
    // methods
    Trainer(const Optimizer& _optimizer, real _lambda, size_t batchSize, size_t nbEpochs, size_t trainSetSize) :
        optimizer(_optimizer),
        lambda(_lambda*batchSize/trainSetSize),
        lambdaBase(_lambda),
        batchSize(batchSize),
        nbEpochs(nbEpochs)
    {};
    
    ~Trainer() {}
    unique_ptr<Trainer> clone() const { return make_unique<Trainer>(*this);}

    string getName() const
    {
        return optimizer.getName() + "_" + getDetail();
    }
    string getDetail() const
    {
        return to_string(lambdaBase) + "_" + to_string(batchSize) + "_" + to_string(nbEpochs);
    }
    
    const auto& getOptimizer() const {return optimizer;}
    auto        getLambda()    const {return lambda;}
    auto        getBatchSize() const {return batchSize;}
    auto        getNbEpochs()  const {return nbEpochs;}
    
private:
    const Optimizer& optimizer;
    real   lambda, lambdaBase;
    size_t batchSize;
    size_t nbEpochs;
};
    
}
#endif /* Trainer_h */
