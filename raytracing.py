import numpy as np
import matplotlib.pyplot as plt

w = 400  # largura da imagem
h = 300  # altura da imagem

def normalizar(x):
    x /= np.linalg.norm(x)
    return x

def intersectar_plano(O, D, P, N):
    # Retorna a distância de O até a interseção do raio (O, D) com o 
    # plano (P, N), ou +inf se não houver interseção.
    # O e P são pontos 3D, D e N (normal) são vetores normalizados.
    denom = np.dot(D, N)
    if np.abs(denom) < 1e-6:
        return np.inf
    d = np.dot(P - O, N) / denom
    if d < 0:
        return np.inf
    return d

def intersectar_esfera(O, D, S, R):
    # Retorna a distância de O até a interseção do raio (O, D) com a 
    # esfera (S, R), ou +inf se não houver interseção.
    # O e S são pontos 3D, D (direção) é um vetor normalizado, R é um escalar.
    a = np.dot(D, D)
    OS = O - S
    b = 2 * np.dot(D, OS)
    c = np.dot(OS, OS) - R * R
    disc = b * b - 4 * a * c
    if disc > 0:
        distSqrt = np.sqrt(disc)
        q = (-b - distSqrt) / 2.0 if b < 0 else (-b + distSqrt) / 2.0
        t0 = q / a
        t1 = c / q
        t0, t1 = min(t0, t1), max(t0, t1)
        if t1 >= 0:
            return t1 if t0 < 0 else t0
    return np.inf

def intersectar(O, D, obj):
    if obj['tipo'] == 'plano':
        return intersectar_plano(O, D, obj['posicao'], obj['normal'])
    elif obj['tipo'] == 'esfera':
        return intersectar_esfera(O, D, obj['posicao'], obj['raio'])

def obter_normal(obj, M):
    # Encontra a normal.
    if obj['tipo'] == 'esfera':
        N = normalizar(M - obj['posicao'])
    elif obj['tipo'] == 'plano':
        N = obj['normal']
    return N
    
def obter_cor(obj, M):
    cor = obj['cor']
    if not hasattr(cor, '__len__'):
        cor = cor(M)
    return cor

def rastrear_raio(rayO, rayD):
    # Encontra o primeiro ponto de interseção com a cena.
    t = np.inf
    for i, obj in enumerate(cena):
        t_obj = intersectar(rayO, rayD, obj)
        if t_obj < t:
            t, obj_idx = t_obj, i
    # Retorna None se o raio não intersectar nenhum objeto.
    if t == np.inf:
        return
    # Encontra o objeto.
    obj = cena[obj_idx]
    # Encontra o ponto de interseção no objeto.
    M = rayO + rayD * t
    # Encontra as propriedades do objeto.
    N = obter_normal(obj, M)
    cor = obter_cor(obj, M)
    toL = normalizar(L - M)
    toO = normalizar(O - M)
    # Sombra: verifica se o ponto está na sombra ou não.
    l = [intersectar(M + N * .0001, toL, obj_sombra) 
            for k, obj_sombra in enumerate(cena) if k != obj_idx]
    if l and min(l) < np.inf:
        return
    # Começa a computar a cor.
    col_ray = ambiente
    # Sombreamento de Lambert (difuso).
    col_ray += obj.get('difuso_c', difuso_c) * max(np.dot(N, toL), 0) * cor
    # Sombreamento de Blinn-Phong (especular).
    col_ray += obj.get('especular_c', especular_c) * max(np.dot(N, normalizar(toL + toO)), 0) ** especular_k * cor_luz
    return obj, M, N, col_ray

def adicionar_esfera(posicao, raio, cor):
    return dict(tipo='esfera', posicao=np.array(posicao), 
        raio=np.array(raio), cor=np.array(cor), reflexao=.5)
    
def adicionar_plano(posicao, normal):
    return dict(tipo='plano', posicao=np.array(posicao), 
        normal=np.array(normal),
        cor=lambda M: (cor_plano0 
            if (int(M[0] * 2) % 2) == (int(M[2] * 2) % 2) else cor_plano1),
        difuso_c=.75, especular_c=.5, reflexao=.25)
    
# Lista de objetos.
cor_plano0 = 1. * np.ones(3)
cor_plano1 = 0. * np.ones(3)
cena = [adicionar_esfera([.75, .1, 1.], .6, [0., 0., 1.]),
         adicionar_esfera([-.75, .1, 2.25], .6, [.5, .223, .5]),
         adicionar_esfera([-2.75, .1, 3.5], .6, [1., .572, .184]),
         adicionar_plano([0., -.5, 0.], [0., 1., 0.]),
    ]

# Posição e cor da luz.
L = np.array([5., 5., -10.])
cor_luz = np.ones(3)

# Parâmetros padrão de luz e material.
ambiente = .05
difuso_c = 1.
especular_c = 1.
especular_k = 50

profundidade_max = 5  # Número máximo de reflexões de luz.
col = np.zeros(3)  # Cor atual.
O = np.array([0., 0.35, -1.])  # Câmera.
Q = np.array([0., 0., 0.])  # Câmera apontando para.
img = np.zeros((h, w, 3))

r = float(w) / h
# Coordenadas da tela: x0, y0, x1, y1.
S = (-1., -1. / r + .25, 1., 1. / r + .25)

# Loop por todos os pixels.
for i, x in enumerate(np.linspace(S[0], S[2], w)):
    if i % 10 == 0:
        print(i / float(w) * 100, "%")
    for j, y in enumerate(np.linspace(S[1], S[3], h)):
        col[:] = 0
        Q[:2] = (x, y)
        D = normalizar(Q - O)
        profundidade = 0
        rayO, rayD = O, D
        reflexao = 1.
        # Loop pelos raios inicial e secundário.
        while profundidade < profundidade_max:
            rastreado = rastrear_raio(rayO, rayD)
            if not rastreado:
                break
            obj, M, N, col_ray = rastreado
            # Reflexão: cria um novo raio.
            rayO, rayD = M + N * .0001, normalizar(rayD - 2 * np.dot(rayD, N) * N)
            profundidade += 1
            col += reflexao * col_ray
            reflexao *= obj.get('reflexao', 1.)
        img[h - j - 1, i, :] = np.clip(col, 0, 1)

plt.imsave('figura.png', img)