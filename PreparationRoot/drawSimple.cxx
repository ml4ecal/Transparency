//
//---- plot Laser
//



// 
//  std::string split implementation by using delimeter as a character.
// 
//  https://thispointer.com/how-to-split-a-string-in-c/ 
// 
std::vector<std::string> split(std::string strToSplit, char delimeter) {
  std::stringstream ss(strToSplit);
  std::string item;
  std::vector<std::string> splittedStrings;
  while (std::getline(ss, item, delimeter))
  {
    splittedStrings.push_back(item);
  }
  return splittedStrings;
}




void drawSimple(std::string nameInputFile = "Laser2017_noTP.root") {
  
  gStyle->SetOptStat(0);
  
  TFile* fileIn = new TFile(nameInputFile.c_str(), "READ");
  TTree* ntu      = (TTree*) fileIn -> Get ("laser");
  
  std::cout << " entries = " << ntu->GetEntries() << std::endl;
  
  TString lumi_string  = Form ("lumi");
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
  
  
  std::string nameInputFile_no_slash;
  std::vector<std::string> v_nameInputFile_no_slash = split(nameInputFile, '/');
  nameInputFile_no_slash = v_nameInputFile_no_slash.at(v_nameInputFile_no_slash.size()-1);
  std::cout << " nameInputFile_no_slash = " << nameInputFile_no_slash << std::endl;
  
  std::string to_save;
  to_save = "plots/cclaser_" + nameInputFile_no_slash + ".png";
  cclaser->SaveAs(to_save.c_str());
  to_save = "plots/cclaser_" + nameInputFile_no_slash + ".root";
  cclaser->SaveAs(to_save.c_str());
 
  
  
  //---- and now the luminosity
  toDraw = Form ("%s:%s", lumi_string.Data(), time_string.Data());  
  std::cout << " toDraw = " << toDraw.Data() << std::endl;
  
  ntu->Draw(toDraw.Data(), toCut.Data(), "goff");
  std::cout << " ntu->GetSelectedRows() = " << ntu->GetSelectedRows() << std::endl;
  
  TGraph *gr_lumi  = new TGraph(ntu->GetSelectedRows(), ntu->GetV2(), ntu->GetV1());  
  
  
  //---- style ----
  
  gr_lumi->SetMarkerSize  (0.2);               
  gr_lumi->SetMarkerStyle (20);              
  gr_lumi->SetMarkerColor (kBlue);            
  gr_lumi->SetLineWidth (1);                 
  gr_lumi->SetLineColor (kBlue);              
  
  //---- style (end) ----
  
  
  TCanvas* cclumi = new TCanvas ("cclumi", toCut.Data(), 1600, 600);
  gr_lumi->SetTitle(toCut.Data());
  gr_lumi->Draw("AP");
  gr_lumi->GetYaxis()->SetTitle("luminosity");
  gr_lumi->GetXaxis()->SetTitle("time");
  gr_lumi->GetXaxis()->SetTimeDisplay(1);
  gr_lumi->GetXaxis()->SetNdivisions(-503);
  gr_lumi->GetXaxis()->SetTimeFormat("%Y-%m-%d %H:%M");
  gr_lumi->GetXaxis()->SetTimeOffset(0,"gmt");
  cclumi->SetGrid();
  
  to_save = "plots/cclumi_" + nameInputFile_no_slash + ".lumi.png";
  cclumi->SaveAs(to_save.c_str());
  to_save = "plots/cclumi_" + nameInputFile_no_slash + ".lumi.root";
  cclumi->SaveAs(to_save.c_str());
  
  
  
  
  
  
}



