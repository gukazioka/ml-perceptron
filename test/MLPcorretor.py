import os
import numpy as np
print("\x1b[2J\x1b[1;1H") 

os.chdir(r'E:\IFTM atual\2021-02\Inteligência Artificial\classificação de digitos\João Pedro - João Paulo - Renato - Thiago')


# Lendo o arquivo de saídas esperadas (target)
t=np.loadtxt('target.csv',delimiter=';',skiprows=0)

#os.chdir(r'E:\IFTM atual\2021-01\Inteligência Artificial\Trabalho 4 - classificação de dígitos MLP\Lucas Luan - Lucas Neto - Paulo Ricardo')
#t=np.loadtxt('targest16.csv',delimiter=';',skiprows=0)
vanterior=np.loadtxt('vnovo.csv')
v0anterior=np.loadtxt('v0novo.csv')
wanterior=np.loadtxt('wnovo.csv')
w0anterior=np.loadtxt('w0novo.csv')




(vent,neur)=np.shape(vanterior)
(vsai,numclasses)=np.shape(t)
limiar=0
zin=np.zeros((1,neur))
target=np.zeros((vsai,1))

os.chdir(r'E:\IFTM atual\2020-02\Inteligência Artificial\digitos')


###################### Limiarização


#### Teste da rede
aminicial=101
amtestedigitos=35
yteste=np.zeros((vsai,1))
k2='_'
k4='.txt'
cont=0
contcerto=0
#ordem=np.zeros(amostras)
for m in range(10):   
    k1=str(m)   
    for n in range(amtestedigitos):      
        k3a=n+aminicial
        k3=str(k3a)
        nome=k1+k2+k3+k4
        xteste=np.loadtxt(nome)
        for m2 in range(vsai):
            for n2 in range(neur):
                zin[0][n2]=np.dot(xteste,vanterior[:,n2])+v0anterior[n2][0]
            z=np.tanh(zin)
            yin=np.dot(z,wanterior)+w0anterior
            y=np.tanh(yin)
        for j in range(vsai):
            if yin[0][j]>=limiar:
                y[0][j]=1.0
            else:
                y[0][j]=-1.0
        for j in range(vsai):
            yteste[j][0]=y[0][j]
        
        for j in range(vsai):
            target[j][0]=t[j][m]
        soma=np.sum(y-target)
        if soma==0:
            contcerto=contcerto+1
        cont=cont+1
taxa=contcerto/cont
print(taxa)        


################## Distância Euclidiana
### Teste da rede
#aminicial=101
#amtestedigitos=35
#yteste=np.zeros((vsai,1))
#k2='_'
#k4='.txt'
#cont=0
#contcerto=0
#
#for m in range(10):   
#    k1=str(m)   
#    for n in range(amtestedigitos):      
#        k3a=n+aminicial
#        k3=str(k3a)
#        nome=k1+k2+k3+k4
#        xteste=np.loadtxt(nome)
#        for m2 in range(vsai):
#            for n2 in range(neur):
#                zin[0][n2]=np.dot(xteste,vanterior[:,n2])+v0anterior[n2]
#            z=np.tanh(zin)
#            yin=np.dot(z,wanterior)+w0anterior
#            y=np.tanh(yin)
#        disteuclidiana=np.zeros((1,numclasses))
#        for j in range(numclasses):
#            distaux=0
#            for m3 in range(vsai):
#                distaux=distaux+(y[0][m3]-t[m3][j])**2
#            disteuclidiana[0][j]=np.sqrt(distaux)
#        indice=disteuclidiana.argmin()
#        if indice==m:
#            contcerto=contcerto+1
#        cont=cont+1
#        
#taxa=contcerto/cont
#print(taxa)        
