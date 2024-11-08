


void merge(){

  Int_t           runNumber;
  Int_t           LS;
  Int_t           time;
  Int_t           eventId;
  Int_t           BX;
  Float_t         energy_EB[61200];
  Float_t         energy_EE[14648];
  

  TFile* fileMerged = new TFile ("merged.root", "RECREATE");
  TTree* mergedTree = new TTree("mergedTree", "");
  
  Float_t         energy_EB_all_prompt;
  Float_t         energy_EB_hlt;
  Float_t         energy_EB_model;
  mergedTree->Branch("energy_EB_all_prompt", &energy_EB_all_prompt, "F");
  mergedTree->Branch("energy_EB_hlt", &energy_EB_hlt, "F");
  mergedTree->Branch("energy_EB_model", &energy_EB_model, "F");
  
  
  TFile *_file0 = TFile::Open("dumpPROMPTRECO.root");

  TFile *_file1 = TFile::Open("dumpAVARSI.root");
  
  TFile *_file2 = TFile::Open("dumpHLT.root");
  
  
  TTree* tree0 = (TTree*) _file0->Get("TreeProducerNoise/tree");

  TTree* tree1 = (TTree*) _file1->Get("TreeProducerNoise/tree");

  TTree* tree2 = (TTree*) _file2->Get("TreeProducerNoise/tree");
  
  
  tree0->SetBranchAddress("energy_EB", energy_EB);
  tree1->SetBranchAddress("energy_EB", energy_EB);
  tree2->SetBranchAddress("energy_EB", energy_EB);
  
  
  int nEntries = tree0->GetEntries();
  
  std::cout << " nEntries = " << nEntries << std::endl;
  
  for (int iEntry = 0; iEntry<nEntries; iEntry++) {
    
    
    tree0->GetEntry(iEntry);
    
    std::cout << " iEntry = " << iEntry << " : " << nEntries << std::endl;
    
    for (int ixtal=0; ixtal<61200; ixtal++) {
      if (energy_EB[ixtal] >0.00001) {
//         std::cout << "   energy_EB[0] = " << energy_EB[0];
        energy_EB_all_prompt = energy_EB[ixtal];
        
        tree1->GetEntry(iEntry);
//         std::cout << "   energy_EB[0] = " << energy_EB[0];
        energy_EB_model = energy_EB[ixtal];
        
        tree2->GetEntry(iEntry);
//         std::cout << "   energy_EB[0] = " << energy_EB[0];
        energy_EB_hlt = energy_EB[ixtal];
        
//         std::cout << std::endl;    
        
        mergedTree->Fill();
        
      }
      
    }
    
    
    
  }
  
  
  fileMerged->cd();
  mergedTree->Write();
  
  
  
}


