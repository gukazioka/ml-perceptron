import numpy as np
import random as rd
from glob import iglob

ENTRADAS = 256

def get_samples(sample_number: int, number: int) -> list:
    samples_qtd = 0
    samples = list()
    for path in iglob(f'data/{number}_*.txt'):
        samples_qtd += 1
        if samples_qtd > sample_number:
            break
        with open(path, 'r') as file:
            samples.append(np.array(list(filter(lambda x: x, file.read().split(' ')))))
    return samples

class MultilayerPerceptron:

    def __init__(
        self,
        neurons: int,
        alpha: float,
        tolerated_error: float,
    ) -> None:
        self.alpha = alpha
        self.neurons = neurons
        self.tolerated_error = tolerated_error
        self.listaciclo = []
        self.listaerro = []
        self.vsai = 10
        self.ampdigitos = 50

    def initialize(self):
        self.samples = self.ampdigitos * self.vsai
        x = np.zeros((self.samples, ENTRADAS))
        ordem = np.zeros(self.samples)

        cont = 0
        for m in range(self.vsai):
            todas_amostras = get_samples(self.ampdigitos, m)

            for entrada in todas_amostras:
                x[cont,:] = entrada[:]
                ordem[cont] = m
                cont += 1
        ordem = ordem.astype('int')

        (amostras, vsai) = np.shape(x)

        t = np.loadtxt('data/targets10.csv', delimiter=';', skiprows=0)
        (vsai, amostras) = np.shape(t)

        vanterior = np.zeros((ENTRADAS, self.neurons))

        for i in range(ENTRADAS):
            for j in range(self.neurons):
                vanterior[i][j] = rd.uniform(-1, 1)
        v0anterior = np.zeros((1, self.neurons))
        for j in range(self.neurons):
            v0anterior[0][j] = rd.uniform(-1, 1)

        wanterior = np.zeros((self.neurons, vsai))
        for i in range(self.neurons):
            for j in range(vsai):
                wanterior[i][j] = rd.uniform(-0.2, 0.2)
        w0anterior = np.zeros((1, vsai))
        for j in range(vsai):
            w0anterior = np.zeros((1, vsai))

        vnovo = np.zeros((ENTRADAS, self.neurons))
        v0novo = np.zeros((1, self.neurons))
        wnovo = np.zeros((self.neurons, vsai))
        w0novo = np.zeros((1, vsai))
        zin = np.zeros((1, self.neurons))
        z = np.zeros((1, self.neurons))
        deltinhak = np.zeros((vsai, 1))
        deltaw0 = np.zeros((vsai, 1))
        deltinha = np.zeros((1, self.neurons))
        xaux = np.zeros((1, ENTRADAS))
        h = np.zeros((vsai, 1))
        target = np.zeros((vsai, 1))
        deltinha2 = np.zeros((self.neurons, 1))
        ciclo = 0
        errototal = 100000

        while self.tolerated_error < errototal:
            errototal = 0
            for padrao in range(amostras):
                for j in range(self.neurons):
                    zin[0][j] = np.dot(
                        x[padrao, :], vanterior[:, j]) + v0anterior[0][j]
                z = np.tanh(zin)
                yin = np.dot(z, wanterior) + w0anterior
                y = np.tanh(yin)

                for m in range(vsai):
                    h[m][0] = y[0][m]
                for m in range(vsai):
                    target[m][0] = t[0][ordem[padrao]]

                errototal = errototal + np.sum(0.5*((target-h)**2))

                deltinhak = (target-h)*(1+h)*(1-h)
                deltaw = self.alpha*(np.dot(deltinhak, z))
                deltaw0 = self.alpha*deltinhak
                deltinhain = np.dot(np.transpose(deltinhak),
                                    np.transpose(wanterior))
                deltinha = deltinhain*(1+z)*(1-z)

                for m in range(self.neurons):
                    deltinha2[m][0] = deltinha[0][m]
                for k in range(ENTRADAS):
                    xaux[0][k] = x[padrao][k]

                deltav = self.alpha*np.dot(deltinha2, xaux)
                deltav0 = self.alpha*deltinha

                vnovo = vanterior+np.transpose(deltav)
                v0novo = v0anterior+np.transpose(deltav0)
                wnovo = wanterior+np.transpose(deltaw)
                w0novo = w0anterior+np.transpose(deltaw0)
                vanterior = vnovo
                v0anterior = v0novo
                wanterior = wnovo
                w0anterior = w0novo

            ciclo = ciclo+1
            self.listaciclo.append(ciclo)
            self.listaerro.append(errototal)

            print('Ciclo\t Erro')
            print(ciclo, '\t', errototal)
        
        np.savetxt('test/vnovo.csv', vnovo, delimiter=';')
        np.savetxt('test/v0novo.csv', v0novo, delimiter=';')
        np.savetxt('test/wnovo.csv', wnovo, delimiter=';')
        np.savetxt('test/w0novo.csv', w0novo, delimiter=';')



if __name__ == '__main__':
    mlp = MultilayerPerceptron(200, 0.005, 0.05)
    mlp.initialize()
