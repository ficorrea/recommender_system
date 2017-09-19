# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.neighbors import NearestNeighbors, DistanceMetric
import sys


#### Manipulação dos datasets ####

def nomear_colunas(start, end):
    """ Nomear colunas das tabelas """
    colunas = ['col' + str(i) for i in range(start, end)]
    return colunas

# Manipulação dataset com lista e gênero de filmes
dataset_item = pd.read_table('u.item', sep='\\|',
                             engine='python',
                             names=nomear_colunas(0, 24))

id_titulo = dataset_item.iloc[:, 0:2]
generos = dataset_item.iloc[:, 5:24]


# Manipulação dataset com avaliações dos filmes
dataset_rating = pd.read_table('u.data', sep='\\\t',
                               engine='python',
                               names=nomear_colunas(0, 4))
data_avaliacao = dataset_rating.iloc[:, 0:3]

# Agrupando dataset de avaliações pela média
aval_media = pd.DataFrame([data_avaliacao.groupby('col1')['col2'].mean()])


def calcular_distancia(metrica, genero_e_avaliacoes):
    """ Calculando distância entre gêneros dos filmes """
    distancia = DistanceMetric.get_metric(metrica)
    if genero_e_avaliacoes == 1:
        geral = pd.merge(generos, aval_media,
                         right_index=True, left_index=True,
                         how='outer')
        dist = distancia.pairwise(geral)
    else:
        dist = distancia.pairwise(generos)
    return dist


def consultar_id_filme(filme):
    """ Consulta de nomes e ID e filmes """
    identificacao = id_titulo['col0']
    lista = id_titulo['col1']
    encontrado = 0
    for i in range(len(identificacao)):
        if filme == lista[i]:
            return identificacao[i]
    if encontrado == 0:
        print('Filme não encontrado!!!')
        sys.exit()


def lista_filmes_recomendados(filmes):
    """ Listar filmes escolhidos """
    filme = id_titulo['col1']
    print('\nFilmes Recomendados: \n')
    for i in range(5):
        print(filme[filmes[i]])


def ajuste_indice(indice, tag_filme):
    """ Eliminação do próprio filme, na lista de recomendados """
    ind = []
    indice = indice.tolist()
    for i in range(6):
        ind.append(indice[0][i])
    for i in range(len(ind)):
        if ind[i] == tag_filme:
            ind.pop(i)
            break
    return ind


# filme = input('Nome filme: ')
filme = 'Batman Forever (1995)'
tag_filme = consultar_id_filme(filme)
distancia_calculada = calcular_distancia('euclidean', 0)
classificador = NearestNeighbors(
    n_neighbors=6).fit(generos, distancia_calculada)
array, indices = classificador.kneighbors(generos.iloc[tag_filme - 1, :])
indices = ajuste_indice(indices, tag_filme - 1)
lista_filmes_recomendados(indices)
