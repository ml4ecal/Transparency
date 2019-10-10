

void drawTurnOnCurve () {

  TFile file_in ("in.root", "READ");    
  
  //---- Test
  float delta_t = 0.05/2./2.;
  Int_t n = 40*2;
  float max_time = delta_t*(n-1);
  Double_t x[100], y[100];
  for (Int_t i=0;i<n;i++) {
    x[i] = i*delta_t;
//     y[i] = 1.*sin(x[i]+0.2);
//     y[i] = 1.;
//     y[i] = exp(-x[i]/50.0);
//     y[i] = 0.02*exp(-x[i]/0.1) + 0.98;
//     y[i] = 0.02*exp(-x[i]/0.1) + 0.98 + (x[i]>0.7) * 0.01 * (exp((x[i]-0.7)/4.)-1.);
    
//     
//     1.00 - 0.98    @ EB
//     
    y[i] = 0.02*exp(-x[i]/0.07) + 0.98 + (x[i]>0.7) * 0.01 * (exp((x[i]-0.7)/3.)-1.);
    
//     
//     0.35 - 0.32    @ EE
//     --> 1.00 - 0.91
    
//     float delta_value_max = 0.09;
//     y[i] = delta_value_max * exp(-x[i]/0.07) + (1.-delta_value_max) + (x[i]>0.7) * 0.02 * (exp((x[i]-0.7)/3.)-1.);
    
  }
  
  std::cout << " max_time = " << max_time << std::endl;
  
  TCanvas* cc_transp = new TCanvas ("cc_transp", "", 800, 600);
  
  TGraph* gr_tr = new TGraph(n,x,y);
  gr_tr->SetMarkerSize(1.);
  gr_tr->SetMarkerStyle(20);
  gr_tr->SetMarkerColor(kRed);
  gr_tr->Draw("APL");
 
  
  
  TCanvas* cc_turn_on = new TCanvas ("cc_turn_on", "", 800, 600);
  
  
  int nbin = 200;
  float min = 0;
  float max = 20;
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

  TLine* vertical_line = new TLine (threshold, 0.0, threshold, 1.2);  
  vertical_line->SetLineColor(kBlue);
  vertical_line->SetLineWidth(2.0);
  vertical_line->Draw();
  
}



