//MVA for signal-bkg discrimination, based on MELA angular variables

#include <cstdlib>
#include <iostream>
#include <map>
#include <string>

#include "TChain.h"
#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TObjString.h"
#include "TSystem.h"
#include "TROOT.h"

#include "TMVA/CrossValidation.h"
#include "TMVA/DataLoader.h"
#include "TMVA/Factory.h"
#include "TMVA/Tools.h"
#include "TMVA/TMVAGui.h"

//#include "T3M_common.h"

using namespace TMVA;

void MVA_MELA_angVariables(){
	
	// Output file
	TFile *fout = new TFile("TMVA_output_1_5tev_50k_200k.root", "RECREATE");
	
	// Get the signal and background trees from TFile source(s);
	
    // SIGNAL tree
	TFile *f_sig = new TFile("./MELA_variables_sig_50k_1_5Tev_DLF.root");
	TTree *signalTree     = (TTree*)f_sig->Get("T");

	//BKG tree
	TFile *f_bkg = new TFile("./MELA_variables_bkg_200k_1_5Tev_DLF.root");
	TTree *bkgTree     = (TTree*)f_bkg->Get("T");
	

    Factory *factory = new Factory("TMVA_new", fout, "");
    DataLoader *dataloader = new DataLoader("dataset_1_5tev_50k_200k");
    


    
    //weights
    Double_t sigWeight1  = 1.0;
    Double_t bkgWeight1 = 1.0;
    
  dataloader->AddSignalTree(signalTree,sigWeight1);
  dataloader->AddBackgroundTree(bkgTree,bkgWeight1);
  
  
  dataloader->AddVariable("mZ1",'D');
  dataloader->AddVariable("mZ2",'D');
  dataloader->AddVariable("cosTheta_star",'D');
  dataloader->AddVariable("cosTheta_1",'D');
  dataloader->AddVariable("cosTheta_2",'D');
  dataloader->AddVariable("phi",'D');
  dataloader->AddVariable("phi_1",'D');
    	
    	
    	//cuts
    	TCut cutS="";
    	TCut cutB="";
    	
    	//dataloader->PrepareTrainingAndTestTree( cutS, cutB,"SplitMode=Random:NormMode=NumEvents:!V" );

      dataloader->PrepareTrainingAndTestTree( cutS,cutB, "nTrain_Signal=30000:nTrain_Background=150000:nTest_Signal=1000:nTest_Background=1000");
	// Booking of MVA methods : BDT
    
    factory->BookMethod( dataloader, TMVA::Types::kBDT, "BDT",
                           "!H:!V:NTrees=1000:MinNodeSize=2.5%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=50" );
    
    
   
    factory->BookMethod(dataloader, TMVA::Types::kMLP, "MLP", "H:!V:NeuronType=tanh:VarTransform=N:NCycles=600:HiddenLayers=N+5:TestRate=5:!UseRegulator" );
    
    factory->BookMethod(dataloader, TMVA::Types::kCuts, "Cuts", "!H:!V:FitMethod=MC:EffSel:SampleSize=200000:VarProp=FSmart" );
 
        // Training the MVA methods
        factory->TrainAllMethods();
        
        // Testing the MVA methods
        factory->TestAllMethods();
        
        // Evaluating the MVA methods
        factory->EvaluateAllMethods();
		
	// Save the output
    fout->Close();
    
    std::cout << "==> Wrote root file: " << fout->GetName() << std::endl;
    std::cout << "==> TMVAClassification is done!" << std::endl;
    
    delete factory;
    delete dataloader;
    
    // Launch the GUI for the root macros
    if (!gROOT->IsBatch()){
        TMVAGui("TMVA_output_1_5tev_50k_200k.root");
    	}

}
