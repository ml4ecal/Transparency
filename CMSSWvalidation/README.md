CMSSW
====

where:

    ~/Cern/Code/ECAL/Transparency/CMSSWvalidation

    
code:

    r00t merge.cxx
    
    
    mergedTree = (TTree*) _file0->Get ("mergedTree")

    mergedTree->Draw("energy_EB_model_hlt/energy_EB_hlt")
    mergedTree->Draw("energy_EB_model_hlt/energy_EB_hlt", "energy_EB_model_hlt/energy_EB_hlt>1")

    
    mergedTreeEE->Draw("energy_EE_model_hlt/energy_EE_hlt")
    mergedTreeEE->Draw("energy_EE_model_hlt/energy_EE_prompt")
    mergedTreeEE->Draw("LS")
    mergedTreeEE->Draw("eventId")


    
    