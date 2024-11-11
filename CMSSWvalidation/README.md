CMSSW
====

where:

    ~/Cern/Code/ECAL/Transparency/CMSSWvalidation

    
code:

    r00t merge.cxx
    
    
    mergedTree = (TTree*) _file0->Get ("mergedTree")

    mergedTree->Draw("energy_EB_model_hlt/energy_EB_hlt")
    mergedTree->Draw("energy_EB_model_hlt/energy_EB_hlt", "energy_EB_model_hlt/energy_EB_hlt>1")

    
    
    