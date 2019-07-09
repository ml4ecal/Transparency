//
//---- plot Laser
//

void drawSimple(std::string nameInputFile = "Laser2017_noTP.root") {
  
  gStyle->SetOptStat(0);
  
  TFile* fileIn = new TFile(nameInputFile.c_str(), "READ");
  TTree* ntu      = (TTree*) fileIn -> Get ("laser");
  
  std::cout << " entries = " << ntu->GetEntries() << std::endl;
  
  TString laser_string  = Form ("transparency");
  TString time_string  = Form ("time");
  TString toDraw = Form ("%s:%s", laser_string.Data(), time_string.Data());
  TString toCut = Form ("1");
  
  std::cout << " toDraw = " << toDraw.Data() << std::endl;
  std::cout << " toCut  = " << toCut.Data()  << std::endl;
  
  ntu->Draw(toDraw.Data(), toCut.Data(), "goff");
  std::cout << " ntu->GetSelectedRows() = " << ntu->GetSelectedRows() << std::endl;
  
  TGraph *gr_laser  = new TGraph(ntu->GetSelectedRows(), ntu->GetV2(), ntu->GetV1());  
  
  
  //---- style ----
  
  gr_laser->SetMarkerSize  (0.2);               
  gr_laser->SetMarkerStyle (20);              
  gr_laser->SetMarkerColor (kRed);            
  gr_laser->SetLineWidth (1);                 
  gr_laser->SetLineColor (kRed);              
  
  //---- style (end) ----
  
  
  TCanvas* cclaser = new TCanvas ("cclaser", toCut.Data(), 1600, 600);
  gr_laser->SetTitle(toCut.Data());
  gr_laser->Draw("AP");
  gr_laser->GetYaxis()->SetTitle("transparency");
  gr_laser->GetXaxis()->SetTitle("time");
  gr_laser->GetXaxis()->SetTimeDisplay(1);
  gr_laser->GetXaxis()->SetNdivisions(-503);
  gr_laser->GetXaxis()->SetTimeFormat("%Y-%m-%d %H:%M");
  gr_laser->GetXaxis()->SetTimeOffset(0,"gmt");
  cclaser->SetGrid();
  
  
  
}



