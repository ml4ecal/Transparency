


void merge(){

  Int_t           runNumber;
  Int_t           LS;
  Int_t           time;
  Int_t           eventId;
  Int_t           BX;
  Float_t         energy_EB[61200];
  Float_t         energy_EE[14648];
  

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
    
    if (energy_EB[0] >0.001) {
      std::cout << "   energy_EB[0] = " << energy_EB[0];
      tree1->GetEntry(iEntry);
      std::cout << "   energy_EB[0] = " << energy_EB[0];
      tree2->GetEntry(iEntry);
      std::cout << "   energy_EB[0] = " << energy_EB[0];
      std::cout << std::endl;      
    }
    
    
    
  }
  
  
}


