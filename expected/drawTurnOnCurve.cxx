

void drawTurnOnCurve () {

  TFile file_in ("in.root", "READ");    
  
  //---- Test
  float delta_t = 0.1;
  Int_t n = 20;
  float max_time = delta_t*(n-1);
  Double_t x[100], y[100];
  for (Int_t i=0;i<n;i++) {
    x[i] = i*delta_t;
//     y[i] = 1.*sin(x[i]+0.2);
//     y[i] = 1.;
    y[i] = exp(-x[i]/5.0);
    
//     
//     1.00 - 0.98
//     
  }
  
  std::cout << " max_time = " << max_time << std::endl;
  
  
  TGraph* gr_tr = new TGraph(n,x,y);
  gr_tr->Draw("AC");
 
  
  
  TCanvas* cc_turn_on = new TCanvas ("cc_turn_on", "", 800, 600);
  
  
  int nbin = 100;
  float min = 0;
  float max = 30;
  float delta_value = (max-min)/nbin;
  
  TH1F* h_turn_on = new TH1F ("h_turn_on", "", nbin, min, max);
  float threshold = 10;
  
  int nEvents = 1000;
  
  for (int ibin = 0; ibin<nbin; ibin++) {
    float value = min + (ibin+0.5)*delta_value;
    for (int iEvent = 0; iEvent < nEvents; iEvent++) {
      
      float time = gRandom->Uniform (max_time);
      float value_smeared = value * gr_tr->Eval(time);
      
//       std::cout << " value_smeared = " << value_smeared << std::endl;
      if ( value_smeared > threshold) {
        h_turn_on->Fill(value);
      }
      
    }
    
  }
  
  h_turn_on->Scale(1./nEvents);
  
  h_turn_on->SetLineWidth(2.);
  h_turn_on->SetLineColor(kRed);
  
  h_turn_on->Draw("histo");
  h_turn_on->GetXaxis()->SetTitle("Energy [GeV]");
  h_turn_on->GetYaxis()->SetTitle("Efficiency");
  
  
}

