import os

iphiRange = (121, 140)
ietaRange = (6, 25)


for iphi in range(iphiRange[0], iphiRange[1]):
  for ieta in range(ietaRange[0], ietaRange[1]):
    
    print "ieta:iphi = [", ieta , ":" , iphi, "]"
    
    #todo = "root -l -q   filterForPublic.cxx\(\\\"Transparency/BlueLaser_2017_rereco_v2_newformat.root\\\"," + str(ieta) + "," + str(iphi) + ",0\)"
    todo = "root -l -q   filterForPublic.cxx\(\\\"Transparency/BlueLaser_2011-2018_newformat.root\\\"," + str(ieta) + "," + str(iphi) + ",0\)"
    print todo
    
    os.system(todo)


