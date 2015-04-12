//
//  Data.h
//  NeuralNetworks
//
//  Created by Thomas Hubert on 11/04/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//

#ifndef NeuralNetworks_Data_h
#define NeuralNetworks_Data_h

#include "includes.h"
#include "MNistParser.h"

class DataContainer
{
public:
    DataContainer() {}
    
    size_t getDataSize() const {return trainData[0].size();}
    
    const std::vector<int>& getTrainLabels() const { return trainLabels;}
    const std::vector<int>& getCrossLabels() const { return crossLabels;}
    const std::vector<int>& getTestLabels()  const { return testLabels;}
    
    const std::vector<std::vector<double> >& getTrainData() const {return trainData;}
    const std::vector<std::vector<double> >& getCrossData() const {return crossData;}
    const std::vector<std::vector<double> >& getTestData()  const {return testData;}
    

    void constructCrossData(size_t fractionSize)
    {
        size_t trainSize = trainLabels.size();
        size_t crossSize = trainSize/fractionSize;
        
        crossLabels.reserve(crossSize);
        crossData.reserve(crossSize);
        
        for (size_t i=0; i<trainSize; i+=fractionSize)
        {
            crossLabels.push_back(trainLabels[i]);
            crossData.push_back  (trainData[i]);
        }
        for (size_t i=0; i<crossSize; i++)
        {
            size_t idx = (crossSize-1-i)*fractionSize;
            trainLabels.erase(trainLabels.begin()+idx);
            trainData.erase  (trainData.begin()+idx);
        }
    }

protected:
    std::vector<int> trainLabels;
    std::vector<int> crossLabels;
    std::vector<int> testLabels;
    
    std::vector<std::vector<double> > trainData;
    std::vector<std::vector<double> > crossData;
    std::vector<std::vector<double> > testData;
};
//
class MNistDataContainer : public DataContainer
{
public:
    MNistDataContainer(std::string trainLabelFN, std::string testLabelFN, std::string trainDataFN, std::string testDataFN, size_t crossFraction=6)
    {
        parseLabels(trainLabelFN, trainLabels);
        parseLabels(testLabelFN , testLabels);
        parseImages(trainDataFN , trainData);
        parseImages(testDataFN  , testData);
        
        constructCrossData(crossFraction);
    }
};

#endif
