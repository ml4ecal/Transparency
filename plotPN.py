import os

#iphiRange = (121, 140)
iphiRange = (131, 140)
ietaRange = (6, 25)


for iphi in range(iphiRange[0], iphiRange[1]):
  for ieta in range(ietaRange[0], ietaRange[1]):
    
    print "ieta:iphi = [", ieta , ":" , iphi, "]"
    
    todo = "root -l -q   drawSimple.cxx\(\\\"Transparency/BlueLaser_2017_rereco_v2_newformat.root.filter." + str(ieta) + "." + str(iphi) + ".0.public.root\\\"\)"
    #todo = "root -l -q   drawSimple.cxx\(\\\"Transparency/BlueLaser_2011-2018_newformat.root.filter." + str(ieta) + "." + str(iphi) + ".0.public.root\\\"\)"
    print todo
    
    os.system(todo)


