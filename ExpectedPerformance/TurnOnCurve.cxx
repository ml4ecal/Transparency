#include"tdrstyle.C"
#include<cmath>

void TurnOnCurveDerivativeEE() {

    gStyle->SetOptStat(0);
  
    setTDRStyle();

    std::ifstream in_file;
    std::vector<float> transparency;
    float x;

    // Load real transparency data
    in_file.open("iring26.txt");
    if (!in_file) {
        std::cout << "error" << std::endl;
        exit(1);
    }

    while (in_file >> x) {
        transparency.push_back(x);
    }

    int transparency_size = transparency.size();

    int nbin = 80;
    float minimo = 29;
    float massimo = 31;
    float threshold = 30;
    float delta_value = (massimo-minimo)/nbin;

    int nEvents = 1000;

    TCanvas* cc_turn_on = new TCanvas ("cc_turn_on", "", 800, 600);
    TH1F *h_correction = new TH1F("h_correction", "", nbin, minimo, massimo);
    TH1F *h_real = new TH1F ("h_real", "", nbin, minimo, massimo);

    TF1* lumi_int_function = new TF1 ("lumi_int_function", "[0]*exp(-[1]*x)+(1-[0])*exp([2]*x)");
    TF1* lumi_inst_function = new TF1 ("lumi_inst_function", "[0]*exp(-[1]*(x-[3]))+(1-[0])*exp([2]*(x-[3]))");

    lumi_int_function->SetParameter(0, 9.93e-1);
    lumi_int_function->SetParameter(1, 3.87e-2);
    lumi_int_function->SetParameter(2, 3.22);

    lumi_inst_function->SetParameter(0, 7.89);
    lumi_inst_function->SetParameter(1, 2.71);
    lumi_inst_function->SetParameter(2, -3.02);

    gRandom->SetSeed();

    float lumi_min = 0.000100052299505594;
    float lumi_max = 0.000390878557887418;
    float delta_lumi = lumi_max - lumi_min;
    float lumi_int, lumi_inst_0, time;

    float weight_correction = 0.;
    float weight_real = 0.;

    float weight = 0.;

    // Fill histograms to get turn on curves
    for (int ibin = 0; ibin < nbin; ibin++) {
        float value = minimo + (ibin+0.5)*delta_value;
        std::cout << ibin << "/" << nbin << std::endl;
        for (int iEvent = 0; iEvent < nEvents; iEvent++) {
            for (int i = 0; i < transparency_size ; i++) {
                if (transparency[i] == 1.) { 
                    lumi_int = 0.;
                    lumi_inst_0 = lumi_min + gRandom->Uniform(delta_lumi);             
                    lumi_int += lumi_inst_0*23;
                    lumi_inst_function->SetParameter(3, lumi_inst_0);
                }
                float lumi_inst = lumi_min + gRandom->Uniform(delta_lumi) ;
                lumi_int += lumi_inst*23;  //lumi section = 23 seconds
                float correction = (lumi_int_function->Eval(lumi_int))*(lumi_inst_function->Eval(lumi_inst));
                float value_smeared = value*transparency[i]/correction;
                float value_smeared_real = value*transparency[i];
                weight += pow(lumi_inst_0/lumi_inst, 1/2);
                if (value_smeared > threshold) {
                    h_correction->Fill(value, pow(lumi_inst_0/lumi_inst, 1/2));
                    weight_correction += pow(lumi_inst_0/lumi_inst, 1/2);
                }
               if (value_smeared_real > threshold) {
                    h_real->Fill(value, pow(lumi_inst_0/lumi_inst, 1/2));
                    weight_real += pow(lumi_inst_0/lumi_inst, 1/2);
                }
            }
        }
    }

    h_real->Scale(1./(h_real->GetBinContent(nbin)));

    h_real->SetLineWidth(2.);
    h_real->SetLineColor(kBlue);
    h_real->SetStats(0);

    h_real->Draw("histo");
    h_real->GetXaxis()->SetTitle("Energy [GeV]");
    h_real->GetYaxis()->SetTitle("Efficiency");

    h_correction->Scale(1./(h_correction->GetBinContent(nbin)));

    h_correction->SetLineWidth(2.);
    h_correction->SetLineColor(kRed);
    h_correction->SetStats(0);

    h_correction->Draw("histo same");
    h_correction->GetXaxis()->SetTitle("Energy [GeV]");
    h_correction->GetYaxis()->SetTitle("Efficiency");

    TLine* vertical_line = new TLine (threshold, 0.0, threshold, 1.1);  
    vertical_line->SetLineColor(kBlack);
    vertical_line->SetLineWidth(1.0);
    vertical_line->Draw();

    TLegend *legend = new TLegend();
    legend->AddEntry(h_real,"Without correction");
    legend->AddEntry(h_correction,"With correction");
    legend->Draw();
    
    // Derivatives of the turn on curves
    Double_t x_real[nbin], y_real[nbin], x_correction[nbin], y_correction[nbin], d_real[nbin], d_correction[nbin];

    for (int i = 1; i < nbin+1; i++) {
        x_real[i] = h_real->GetBinCenter(i);
        y_real[i] = h_real->GetBinContent(i);

        x_correction[i] = h_correction->GetBinCenter(i);
        y_correction[i] = h_correction->GetBinContent(i);
    }

    for (int i = 1; i < nbin+1; i++) {
        d_real[i] = (y_real[i]-y_real[i-1])/(x_real[i]-x_real[i-1]);
        d_correction[i] = (y_correction[i]-y_correction[i-1])/(x_correction[i]-x_correction[i-1]);
    }

    Double_t point_real[nbin-8], point_correction[nbin-8];
    double sum_real = 0.;
    double sum_correction = 0.;

    for (int i = 2; i < nbin; i++) {
        sum_real = 0.;
        sum_correction = 0.;
        sum_real = d_real[i-1]+d_real[i]+d_real[i+1];
        sum_correction = d_correction[i-1]+d_correction[i]+d_correction[i+1];
        point_real[i] = sum_real/3;
        point_correction[i] = sum_correction/3;
    }


    TCanvas *c_point = new TCanvas("c_point", "", 800, 700);
    TGraph *real_point = new TGraph(nbin, x_real, point_real);
    TGraph *correction_point = new TGraph(nbin, x_correction, point_correction);
    TMultiGraph *mg2 = new TMultiGraph();

    real_point->SetMarkerSize(1.);
    real_point->SetMarkerStyle(20);
    real_point->SetMarkerColor(kBlue);
    real_point->SetLineColor(kBlue);
    real_point->SetLineWidth(2);
    mg2->Add(real_point);

    correction_point->SetMarkerSize(1.);
    correction_point->SetMarkerStyle(20);
    correction_point->SetMarkerColor(kRed);
    correction_point->SetLineColor(kRed);
    correction_point->SetLineWidth(2);
    mg2->Add(correction_point);

    mg2->Draw("AP");
    mg2->GetXaxis()->SetLimits(29,31);
    mg2->SetMinimum(-1e-3);
    mg2->GetXaxis()->SetTitle("Energy [GeV]");
    mg2->GetXaxis()->SetLabelSize(.045);
    mg2->GetYaxis()->SetTitle("Efficiency derivative [GeV^{-1}]");

    TLegend *p_legend = new TLegend();
    p_legend->AddEntry(real_point,"Without correction");
    p_legend->AddEntry(correction_point,"With correction");
    p_legend->Draw();

    sum_real = 0.;
    sum_correction = 0.;

    for (int i = 1; i < nbin+1; i++) {
        sum_real += d_real[i];
        sum_correction += d_correction[i];
    }
    double mean_real = sum_real/nbin;
    double mean_correction = sum_correction/nbin;

    // Variance computation
    double var_real = 0.;
    double var_correction = 0.;
    weight_real = 0.;
    weight_correction = 0.;
    for (int i = 1; i < nbin+1; i++) {
        var_real += point_real[i]*(x_real[i]-30.)*(x_real[i]-30.);
        weight_real += point_real[i];

        var_correction += point_correction[i]*(x_correction[i]-30.)*(x_correction[i]-30.);
        weight_correction += point_correction[i];
    }
    var_real = var_real/weight_real;
    var_correction = var_correction/weight_correction;
    std::cout << var_real << std::endl;
    std::cout << var_correction << std::endl;

}
