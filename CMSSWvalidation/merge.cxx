


void merge(){

  Int_t           runNumber;
  Int_t           LS;
  Int_t           time;
  Int_t           eventId;
  Int_t           BX;
  Float_t         energy_EB[61200];
  Float_t         energy_EE[14648];
  
  Float_t         energy_EB_1[61200];
  Float_t         energy_EE_1[14648];
  
  Float_t         energy_EB_2[61200];
  Float_t         energy_EE_2[14648];
  
  Float_t         energy_EB_3[61200];
  Float_t         energy_EE_3[14648];
  
  

//   TFile* fileMerged = new TFile ("merged.root", "RECREATE");
//   TFile* fileMerged = new TFile ("merged_2.root", "RECREATE");
//   TFile* fileMerged = new TFile ("merged_3.root", "RECREATE");
  TFile* fileMerged = new TFile ("merged_4.root", "RECREATE");
  TTree* mergedTree = new TTree("mergedTree", "");

//   TFile* fileMergedEE = new TFile ("merged_2_EE.root", "RECREATE");
//   TFile* fileMergedEE = new TFile ("merged_3_EE.root", "RECREATE");
  TFile* fileMergedEE = new TFile ("merged_4_EE.root", "RECREATE");
  TTree* mergedTreeEE = new TTree("mergedTreeEE", "");
  
  float     energy_EB_prompt;
  float     energy_EB_hlt;
  float     energy_EB_model;
  float     energy_EB_model_hlt;
  float     energy_EE_prompt;
  float     energy_EE_hlt;
  float     energy_EE_model;
  float     energy_EE_model_hlt;
  
  mergedTree->Branch("time", &time, "I");
  mergedTree->Branch("LS", &LS, "I");
  mergedTree->Branch("eventId", &eventId, "I");
  mergedTree->Branch("energy_EB_prompt", &energy_EB_prompt, "F");
  mergedTree->Branch("energy_EB_hlt", &energy_EB_hlt, "F");
  mergedTree->Branch("energy_EB_model", &energy_EB_model, "F");
  mergedTree->Branch("energy_EB_model_hlt", &energy_EB_model_hlt, "F");
  
  mergedTreeEE->Branch("time", &time, "I");
  mergedTreeEE->Branch("LS", &LS, "I");
  mergedTreeEE->Branch("eventId", &eventId, "I");
  mergedTreeEE->Branch("energy_EE_prompt", &energy_EE_prompt, "F");
  mergedTreeEE->Branch("energy_EE_hlt", &energy_EE_hlt, "F");
  mergedTreeEE->Branch("energy_EE_model", &energy_EE_model, "F");
  mergedTreeEE->Branch("energy_EE_model_hlt", &energy_EE_model_hlt, "F");

  
//   TFile *_file0 = TFile::Open("dumpPROMPTRECO.root");
// 
//   TFile *_file1 = TFile::Open("dumpAVARSI.root");
//   
//   TFile *_file2 = TFile::Open("dumpHLT.root");
//   
//   TFile *_file3 = TFile::Open("dumpAVARSI_HLT.root");
//   
  

//   TFile *_file0 = TFile::Open("dumpPROMPTRECO_2.root");
//   
//   TFile *_file1 = TFile::Open("dumpAVARSI_2.root");
//   
//   TFile *_file2 = TFile::Open("dumpHLT_2.root");
//   
//   TFile *_file3 = TFile::Open("dumpAVARSI_HLT_2.root");
  
  
  
//   TFile *_file0 = TFile::Open("dumpPROMPTRECO_3.root");
//   
//   TFile *_file1 = TFile::Open("dumpAVARSI_3.root");
//   
//   TFile *_file2 = TFile::Open("dumpHLT_3.root");
//   
//   TFile *_file3 = TFile::Open("dumpAVARSI_HLT_3.root");
  
  
//   TFile *_file0 = TFile::Open("data/dumpPROMPTRECO_3.root");
//   
//   TFile *_file1 = TFile::Open("data/dumpAVARSI_3.root");
//   
//   TFile *_file2 = TFile::Open("data/dumpHLT_3.root");
//   
//   TFile *_file3 = TFile::Open("data/dumpAVARSI_HLT_3.root");
  
  
  
  TFile *_file0 = TFile::Open("data/dumpPROMPTRECO_4.root");
  
  TFile *_file1 = TFile::Open("data/dumpAVARSI_4.root");
  
  TFile *_file2 = TFile::Open("data/dumpHLT_4.root");
  
  TFile *_file3 = TFile::Open("data/dumpAVARSI_HLT_4.root");

  
  
  TTree* tree0 = (TTree*) _file0->Get("TreeProducerNoise/tree");

  TTree* tree1 = (TTree*) _file1->Get("TreeProducerNoise/tree");

  TTree* tree2 = (TTree*) _file2->Get("TreeProducerNoise/tree");

  TTree* tree3 = (TTree*) _file3->Get("TreeProducerNoise/tree");
  
  
  tree0->SetBranchAddress("energy_EB", energy_EB);
  tree1->SetBranchAddress("energy_EB", energy_EB_1);
  tree2->SetBranchAddress("energy_EB", energy_EB_2);
  tree3->SetBranchAddress("energy_EB", energy_EB_3);

  tree0->SetBranchAddress("energy_EE", energy_EE);
  tree1->SetBranchAddress("energy_EE", energy_EE_1);
  tree2->SetBranchAddress("energy_EE", energy_EE_2);
  tree3->SetBranchAddress("energy_EE", energy_EE_3);

  
  tree0->SetBranchAddress("LS", &LS);
  tree0->SetBranchAddress("time", &time);
  tree0->SetBranchAddress("eventId", &eventId);
  

  int minXtalEB = 0;
  int maxXtalEB = 50;
//   int maxXtalEB = 61200;
  
  int nEntries = tree0->GetEntries();
  
  std::cout << " nEntries = " << nEntries << std::endl;
  
  for (int iEntry = 0; iEntry<nEntries; iEntry++) {
    
    
    tree0->GetEntry(iEntry);
    
    std::cout << " iEntry = " << iEntry << " : " << nEntries << std::endl;
    
    for (int ixtal=minXtalEB; ixtal<maxXtalEB; ixtal++) {
      if (energy_EB[ixtal] >0.00001) {
        energy_EB_prompt = energy_EB[ixtal];
//         std::cout << "   energy_EB_prompt[" << ixtal << "] = " << energy_EB[ixtal];
        
        tree1->GetEntry(iEntry);
        energy_EB_model = energy_EB_1[ixtal];
//         std::cout << "   energy_EB_model[" << ixtal << "] = " << energy_EB[ixtal];
        
        tree2->GetEntry(iEntry);
        energy_EB_hlt = energy_EB_2[ixtal];
//         std::cout << "   energy_EB_hlt[" << ixtal << "] = " << energy_EB[ixtal];
        
        tree3->GetEntry(iEntry);
        energy_EB_model_hlt = energy_EB_3[ixtal];
//         std::cout << "   energy_EB_model_hlt[" << ixtal << "] = " << energy_EB[ixtal];
        
        
//         std::cout << " LS = " << LS ;
//         std::cout << std::endl;    
        
        mergedTree->Fill();
        
      }
    }
      
      for (int ixtal=0; ixtal<14648; ixtal++) {
        if (energy_EE[ixtal] >0.00001) {
          energy_EE_prompt = energy_EE[ixtal];
          //         std::cout << "   energy_EE_prompt[" << ixtal << "] = " << energy_EE[ixtal];
          
          tree1->GetEntry(iEntry);
          energy_EE_model = energy_EE_1[ixtal];
          //         std::cout << "   energy_EE_model[" << ixtal << "] = " << energy_EE[ixtal];
          
          tree2->GetEntry(iEntry);
          energy_EE_hlt = energy_EE_2[ixtal];
          //         std::cout << "   energy_EE_hlt[" << ixtal << "] = " << energy_EE[ixtal];
          
          tree3->GetEntry(iEntry);
          energy_EE_model_hlt = energy_EE_3[ixtal];
          //         std::cout << "   energy_EE_model_hlt[" << ixtal << "] = " << energy_EE[ixtal];
          
          
          //         std::cout << " LS = " << LS ;
          //         std::cout << std::endl;    
          
          mergedTreeEE->Fill();
          
        }
        
      }

      
      
    
  }
  
  
  fileMerged->cd();
  mergedTree->Write();
  
  fileMergedEE->cd();
  mergedTreeEE->Write();
  
  
  
}


