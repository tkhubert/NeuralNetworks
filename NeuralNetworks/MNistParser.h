//
//  MNistParser.h
//  NeuralNetworks
//
//  Created by Thomas Hubert on 11/04/2015.
//  Copyright (c) 2015 Thomas Hubert. All rights reserved.
//

#ifndef NeuralNetworks_MNistParser_h
#define NeuralNetworks_MNistParser_h

#include "includes.h"

inline void parseLabels(std::string lblFileName, std::vector<int>& labels)
{
    std::ifstream file(lblFileName, std::ios::in | std::ios::binary);
    if (file.bad() || file.fail())
        throw "failed to open file:" + lblFileName;
    
    uint32_t magicNumber, nbItems;
    
    file.read((char*) &magicNumber, 4);
    file.read((char*) &nbItems    , 4);
    
    if (magicNumber != 0x00000801 || nbItems <= 0)
        throw "MNIST label-file format error";
    
    for (size_t i=0; i<nbItems; ++i)
    {
        uint8_t label;
        file.read((char*) &label, 1);
        labels.push_back((int) label);
    }
}
//
inline void parseImage (std::ifstream& file, std::vector<double>& img, int nbRows, int nbCols, double scaleMin, double scaleMax)
{
    std::vector<uint8_t> imgVec(nbRows*nbCols);
    file.read((char *) &imgVec[0], nbRows*nbCols);
    
    img.resize(nbRows*nbCols);
    for (size_t i=0; i<nbRows; ++i)
        for (size_t j=0; j<nbCols; ++j)
            img[i*nbCols+j] = (imgVec[i*nbCols+j]/255.)*(scaleMax-scaleMin)+scaleMin;
}
//
inline void parseImages(std::string imgFileName, std::vector<std::vector<double> >& images, double scaleMin=0, double scaleMax=1.)
{
    std::ifstream file(imgFileName, std::ios::in | std::ios::binary);
    
    if (file.bad() || file.fail())
        throw "failed to open file:" + imgFileName;
    
    uint32_t magicNumber, nbItems, nbRows, nbCols;
    file.read((char*) &magicNumber, 4);
    file.read((char*) &nbItems    , 4);
    file.read((char*) &nbRows     , 4);
    file.read((char*) &nbCols     , 4);
    
    images.resize(nbItems);
    for (size_t i=0; i<nbItems; i++)
        parseImage(file, images[i], nbRows, nbCols, scaleMin, scaleMax);
}

#endif
