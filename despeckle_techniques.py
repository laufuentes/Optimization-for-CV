import numpy as np 
import matplotlib.pyplot as plt 
import pywt
from skimage import exposure, img_as_float
from despeckle_main import Speckle_removal

def sigma_l2():
      global D 
      return (np.median(np.abs(D))/0.6745)**2

class Other_approachs(Speckle_removal): 
   def Hard_thresholding(t): 
        global H
        res = H.copy()
        for i in range(H.shape[0]): 
            for j in range(H.shape[1]):
                if np.abs(H[i,j, 0])<t: 
                    res[i,j] = np.zeros_like(H[i,j])
        return res
   
   def Soft_thresholding(t):
        global H 
        res = np.zeros_like(H)
        for i in range(H.shape[0]): 
            for j in range(H.shape[1]):
                if np.abs(H[i,j, 0])>=t: 
                    res[i,j] = np.sign(H[i,j])*(np.abs(H[i,j])-t)
        return res
   
   def XT(tau):
        global H, nl
        res_wav = H.copy()
        inf_tau = np.where(np.abs(H) < tau)
        for i in inf_tau: res_wav[i]  = H[i]*np.exp(nl * (np.abs(H[i]) - tau))
        return res_wav
   
   def VisuShrink():
        global log_img
        I = log_img
        return np.std(I)*np.sqrt(2*np.log(log_img.shape[0]*log_img.shape[1]))
   
   def BayesShrink():
        global H, D
        return sigma_l2() / np.std(H)
   
   def thresholding_fct():
        global H, log_img, D
        sigmai = np.var(log_img)
        beta = np.sqrt(2*np.log(log_img.shape[0]*log_img.shape[1]))
        tau =  2*beta*np.abs(sigmai - sigma_l2()) / np.sqrt(sigmai)
        return tau
   
   def run(self, level1, level2, wav,thrmeth=BayesShrink, thrfct=Soft_thresholding):
        global nl, D, H, log_img 
        log_img = self.img_log
        tot = [level1, level2]
        new = []
        for i,l in enumerate(tot): 
            global nl
            if i ==0: nl=1
            else: nl=0.5
            leveln = []
            for i, c in enumerate(l[1:]):
                global D, H, A
                A = l[0]
                D = l[3]
                H = c
                leveln.append(thrfct(thrmeth()))
            new.append(leveln)

        leveln1 = new[0]
        leveln2 = new[1]
        An1 = pywt.idwt2((tot[1][0], (leveln2[0], leveln2[1], leveln2[2])), wav)[:, :tot[0][0].shape[1], :3]
        imglog = pywt.idwt2((An1, (leveln1[0], leveln1[1], leveln1[2])), wav)
        final_img = exposure.rescale_intensity(np.exp(imglog[: , :self.img.shape[1], :3]), out_range=(0,1))
        return final_img



if __name__=='__main__':
    img_path = "/Users/laurafuentesvicente/M2 Maths&IA/Optimization for CV/Project/Dataset_BUSI_with_GT/benign/benign (341).png" #109, 423, 434, 421, 392, 341, 313
    img = plt.imread(img_path)