//
//---- plot Laser
//

void filterForPublic(std::string nameInputFile = "Laser2017_noTP.root", int ix = 62, int iy = 50,  int iz = 1) {
  
  std::cout << " ix = " << ix << std::endl;
  std::cout << " iy = " << iy << std::endl;
  std::cout << " iz = " << iz << std::endl;
  
  gStyle->SetOptStat(0);
  
  TFile* fileIn = new TFile(nameInputFile.c_str(), "READ");
  TTree* ntu      = (TTree*) fileIn -> Get ("ntu");
  
  std::cout << " entries = " << ntu->GetEntries() << std::endl;
  
  TString laser_string  = Form ("nrv");
  TString time_string  = Form ("time[0]");
  TString lumi_string  = Form ("lumi");
  TString fill_time_string  = Form ("fill_time");
  
  
  
//   TString toDraw = Form ("%s:%s:%s:%s", laser_string.Data(), time_string.Data(), lumi_string.Data(), fill_time_string.Data());
  TString toDraw = Form ("%s:%s:%s:(time[0]-fill_time)", laser_string.Data(), time_string.Data(), lumi_string.Data());
  
  //---- B-on
  
//   TString toCut = Form ("ix==%d && iy==%d && iz==%d && bfield > 3.5 ", ix, iy, iz);
  
  TString toCut = Form ("ix==%d && iy==%d && iz==%d && bfield > 3.5 && time[0]>1495000000 && time[0]<1515000000", ix, iy, iz);
  
  //----> 2017: 1495 - 1515 x 10^6
  
  
  std::cout << " toDraw = " << toDraw.Data() << std::endl;
  std::cout << " toCut  = " << toCut.Data()  << std::endl;
  
  std::string name_output_file;
  name_output_file = nameInputFile + ".filter." + std::to_string(ix) + "." + std::to_string(iy) + "." + std::to_string(iz) + ".public.root";
  std::cout << " name_output_file = " << name_output_file << std::endl;
  
  ntu->Draw(toDraw.Data(), toCut.Data(), "goff");
  
//   
//   ntu->GetV1() --> transparency
//   ntu->GetV2() --> time
//   ntu->GetV3() --> lumi
//   ntu->GetV4() --> (time[0]-fill_time)   ---> time since the beginning of the fill
//   
  
  int netries = ntu->GetSelectedRows();
  
  TFile* fileOut = new TFile (name_output_file.c_str(), "RECREATE");   
  TTree *outputTree = new TTree ("laser", "laser");
  float transparency;
  int time;
  float lumi;
  float fill_time;
  float time_in_fill;
  
  outputTree->Branch("transparency", &transparency, "transparency/F");
  outputTree->Branch("time", &time, "time/I");
  outputTree->Branch("lumi", &lumi, "lumi/F");
//   outputTree->Branch("fill_time", &fill_time, "fill_time/F");
  outputTree->Branch("time_in_fill", &time_in_fill, "time_in_fill/F");
  
  int entries = netries;
  std::cout << " netries = " << netries << std::endl;
  for (int iEntry=0; iEntry<netries; iEntry++) {
    if (!(iEntry%50000)) std::cout << "   " << iEntry << " :: " << netries << std::endl;
    transparency = ntu->GetV1()[iEntry];
    time = ntu->GetV2()[iEntry];
    lumi = ntu->GetV3()[iEntry];
//     fill_time = ntu->GetV4()[iEntry];
    time_in_fill = ntu->GetV4()[iEntry];
    
    outputTree->Fill();       
  }
 
 outputTree->AutoSave();
 fileOut->Close();
 
}




