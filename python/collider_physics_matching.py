#To  run  this macro you need  python3 + root
#that are provided in the environment that we setup during the class  with
#conda activate myenv.
#However if you have any version of root compiled with python3,
#you can skip the activation  of the environment

import ROOT
from ROOT import RDataFrame
from ROOT import TChain, TSelector, TTree, TH1F, TCanvas, TFile, TEfficiency, TLegend
from ROOT import TLorentzVector
from ROOT import TVector3
from array import array
import numpy as np
ROOT.gROOT.SetBatch(ROOT.kTRUE)
file ="../data/ntuple_mumuHZZ4mu_final.root"
f = TFile(file)  #open the rootfile
tree = f.Get("T") #load the tree
N = tree.GetEntries()
print("total number of events=",N)

hGenMu1Pt = TH1F("hGenMu1Pt","Gen Muon 1 Pt",200,0,200)
hGenMu2Pt = TH1F("hGenMu2Pt","Gen Muon 2 Pt",200,0,200)
hGenMu3Pt = TH1F("hGenMu3Pt","Gen Muon 3 Pt",200,0,200)
hGenMu4Pt = TH1F("hGenMu4Pt","Gen Muon 4 Pt",200,0,200)

hRecoMu1Pt = TH1F("hRecoMu1Pt","Reco Muon 1 Pt",200,0,200)
hRecoMu2Pt = TH1F("hRecoMu2Pt","Reco Muon 2 Pt",200,0,200)
hRecoMu3Pt = TH1F("hRecoMu3Pt","Reco Muon 3 Pt",200,0,200)
hRecoMu4Pt = TH1F("hRecoMu4Pt","Reco Muon 4 Pt",200,0,200)

histoNum_Pt = TH1F("histoNum_Pt","Gen Muon Pt after matching with reco",200,0,200)
histoDen_Pt = TH1F("histoDen_Pt","Gen Muon Pt before matching with reco",200,0,200)
hPtRes  = TH1F("hPtRes","hPtRes", 100,-0.5,0.5)

hGenMu1Pt.Reset()
hGenMu2Pt.Reset()
hGenMu3Pt.Reset()
hGenMu4Pt.Reset()
hRecoMu1Pt.Reset()
hRecoMu2Pt.Reset()
hRecoMu3Pt.Reset()
hRecoMu4Pt.Reset()
histoNum_Pt.Reset()
histoDen_Pt.Reset()

#for i in range(N):  #to loop on all the events in the rootfile
nevt=1000
for i in range(nevt):
    tree.GetEntry(i)
    print("----- event", i)
    n_genpart = tree.nmcp
    print("ngenpart=",n_genpart)
    GenMuonList = []
    RecoMuonList = []
    for pdg, st, ene, px, py, pz in zip(tree.mcpdg, tree.mcgst, tree.mcene, tree.mcmox, tree.mcmoy, tree.mcmoz):
            if(abs(pdg)==13 and st==1):  #only stable muons produced after muon beam scattering
                #print("sim mu energy=", ene)
                genMuonVector = TLorentzVector()  #create a TLorentzVector
                genMuonVector.SetPxPyPzE(px,py,pz,ene) #fill it with stable muons 4-momentum coordinates
                print("gen mu pt=", genMuonVector.Pt())  #the method Pt() does all the calculations
                GenMuonList.append(genMuonVector)   # create a list of TLorentz vector
                #print("stored muons",len(GenMuonList))
                if(len(GenMuonList)==4):
                    HiggsMC = TLorentzVector()
                    HiggsMC = GenMuonList[0] + GenMuonList[1] + GenMuonList[2] + GenMuonList[3]
                    #print("H mass=",HiggsMC.M())
                    MuonMC_ptList = []
                    MuonMC_ptList.append(GenMuonList[0].Pt())
                    MuonMC_ptList.append(GenMuonList[1].Pt())
                    MuonMC_ptList.append(GenMuonList[2].Pt())
                    MuonMC_ptList.append(GenMuonList[3].Pt())
                    #MuonMC_ptList.sort()
                    hGenMu1Pt.Fill(MuonMC_ptList[0])
                    hGenMu2Pt.Fill(MuonMC_ptList[1])
                    hGenMu3Pt.Fill(MuonMC_ptList[2])
                    hGenMu4Pt.Fill(MuonMC_ptList[3])
                                    
                                    
    for id, ene, px, py, pz in zip(tree.rctyp, tree.rcene, tree.rcmox, tree.rcmoy, tree.rcmoz):
        if(abs(id)==13):
            #print("reco mu ene=", ene)
            recoMuonVector = TLorentzVector()
            recoMuonVector.SetPxPyPzE(px,py,pz,ene)
            print("reco mu pt=", recoMuonVector.Pt())
            RecoMuonList.append(recoMuonVector)
            #print("stored reco muons",len(RecoMuonList))
            if(len(RecoMuonList)==4):
                HiggsReco = TLorentzVector()
                HiggsReco = RecoMuonList[0] + RecoMuonList[1] + RecoMuonList[2] + RecoMuonList[3];
                #print("Reco H mass=",HiggsReco.M())
                MuonReco_ptList = []
                MuonReco_ptList.append(RecoMuonList[0].Pt())
                MuonReco_ptList.append(RecoMuonList[1].Pt())
                MuonReco_ptList.append(RecoMuonList[2].Pt())
                MuonReco_ptList.append(RecoMuonList[3].Pt())
                #MuonReco_ptList.sort()
                hRecoMu1Pt.Fill(MuonReco_ptList[0])
                hRecoMu2Pt.Fill(MuonReco_ptList[1])
                hRecoMu3Pt.Fill(MuonReco_ptList[2])
                hRecoMu4Pt.Fill(MuonReco_ptList[3])
    for k in range(len(GenMuonList)):
        histoDen_Pt.Fill(GenMuonList[k].Pt());
        dR = 1000
        for i in range(len(RecoMuonList)):
            thisDR=RecoMuonList[i].DrEtaPhi(GenMuonList[k])
            #print("thisdR=", thisDR)
            if(thisDR<dR):
                dR=thisDR
                if(dR<0.01):
                    histoNum_Pt.Fill(GenMuonList[k].Pt());
                    pt_res = (GenMuonList[k].Pt()-RecoMuonList[i].Pt())/GenMuonList[k].Pt()
                    hPtRes.Fill(pt_res)

c1 = TCanvas( 'c1', 'Gen Muon Pt', 200, 10, 700, 500 )
hGenMu1Pt.SetLineColor(23)
hGenMu2Pt.SetLineColor(4)
hGenMu3Pt.SetLineColor(8)
hGenMu4Pt.SetLineColor(2)
hGenMu1Pt.Draw()
hGenMu2Pt.Draw("same")
hGenMu3Pt.Draw("same")
hGenMu4Pt.Draw("same")
hGenMu1Pt.SetTitle(" ");
hGenMu1Pt.GetXaxis().SetTitle("Gen Muon p_{T} [GeV/c]");
hGenMu1Pt.GetYaxis().SetTitle("Number of muons");
legend = TLegend(0.6,0.8,0.85,0.4);
legend.SetHeader("Gen Muon p_{T} ","C")
legend.AddEntry(hGenMu1Pt,"p_{T}^{#mu_{1}}","l");
legend.AddEntry(hGenMu2Pt,"p_{T}^{#mu_{2}}","l");
legend.AddEntry(hGenMu3Pt,"p_{T}^{#mu_{3}}","l");
legend.AddEntry(hGenMu4Pt,"p_{T}^{#mu_{4}}","l");
ROOT.gStyle.SetLegendTextSize(0.05);
ROOT.gStyle.SetOptStat("e");
legend.Draw();
c1.Draw()
c1.SaveAs("./GenMuon_Pt.png")

c2 = TCanvas( 'c2', 'Reco Muon Pt', 200, 10, 700, 500 )
hRecoMu1Pt.SetLineColor(2)
hRecoMu2Pt.SetLineColor(4)
hRecoMu3Pt.SetLineColor(8)
hRecoMu4Pt.SetLineColor(23)
hRecoMu1Pt.Draw()
hRecoMu2Pt.Draw("same")
hRecoMu3Pt.Draw("same")
hRecoMu4Pt.Draw("same")
hRecoMu1Pt.GetXaxis().SetTitle("Reco Muon p_{T} [GeV/c]");
hRecoMu1Pt.GetYaxis().SetTitle("Number of muons");
c2.Draw()
c2.SaveAs("./RecoMuon_Pt.png")

c3 = TCanvas( 'c3', 'Efficiency vs Pt', 200, 10, 700, 500 )
c3.cd()
h_EffPt=histoNum_Pt.Clone();
h_EffPt.Divide(histoDen_Pt);
h_EffPt.Draw("h");
#h_EffPt.Draw("he");
c3.Draw()
c3.SaveAs("./Eff_Muon_Pt.png")

#c5 = TCanvas( 'c5', 'Efficiency vs Pt with TEfficiency', 200, 10, 700, 500 )
#c5.cd();
#genReco_Eff_pT = TEfficiency(histoNum_Pt,histoDen_Pt)
#genReco_Eff_pT.Draw()
#c5.Draw()

#c6 = TCanvas( 'c6', 'Efficiency vs Pt with TEfficiency', 200, 10, 700, 500 )
#c6.cd();
#hPtRes.Draw()
#c6.Draw()
