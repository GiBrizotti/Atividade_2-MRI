import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


# FUNÇÕES QUE DEFINEM VARIÁVEIS E CARREGA DADOS 

def carregar_dados_mat(caminho_arquivo, nome_variavel):
    # Carrega o arquivo .mat contendo os dados do espaço-k.
    # Retorna a matriz tridimensional (N_leitura, N_fase, N_fatias). 

    dados_mat = loadmat(caminho_arquivo)
    dados_kspace = dados_mat[nome_variavel]
    return dados_kspace


def definir_parametros_estudo():
    # Retorna as configurações da análise.
    
    configuracoes = {
        'fatias_q1': [6, 10, 14, 18, 22], # Fatias pedidas na Q1 (base 1)
        'fatia_q2': 13,                    # Fatia pedida na Q2 (base 1)
        'limiar_spike_std': 10             # Limiar de desvios padrões para achar spike
    }
    return configuracoes





# FUNÇÕES DE PROCESSAMENTO E RECONSTRUÇÃO 

def reconstruir_imagem_2d(espaco_k_2d):
    # Aplica a IFFT 2D nos dados do espaço-k para gerar a imagem anatômica.

    espaco_k_shifted = np.fft.ifftshift(espaco_k_2d)
    imagem_complexa = np.fft.ifft2(espaco_k_shifted)
    imagem_magnitude = np.abs(imagem_complexa)
    return imagem_magnitude


def subamostrar_espaco_k(espaco_k_2d, fator_r):
    # Sub-amostra o espaço-k mantendo 1 linha a cada 'fator_r' linhas na direção de codificação de fase.

    espaco_k_sub = np.zeros_like(espaco_k_2d)
    espaco_k_sub[::fator_r, :] = espaco_k_2d[::fator_r, :]
    return espaco_k_sub


def remover_spike_espaco_k(espaco_k_2d, limiar_std=10):
    # Detecta o spike no espaço-k e substitui o valor pela média dos pixels vizinhos.

    espaco_k_corrigido = espaco_k_2d.copy()
    magnitude_k = np.abs(espaco_k_corrigido)
    
    media = np.mean(magnitude_k)
    desvio_padrao = np.std(magnitude_k)
    
    linhas_spike, colunas_spike = np.where(magnitude_k > (media + limiar_std * desvio_padrao))
    
    for r, c in zip(linhas_spike, colunas_spike):
        r_min, r_max = max(0, r - 1), min(espaco_k_2d.shape[0], r + 2)
        c_min, c_max = max(0, c - 1), min(espaco_k_2d.shape[1], c + 2)
        
        vizinhanca = espaco_k_corrigido[r_min:r_max, c_min:c_max].copy()
        mascara_spike = np.ones(vizinhanca.shape, dtype=bool)
        mascara_spike[r - r_min, c - c_min] = False
        
        espaco_k_corrigido[r, c] = np.mean(vizinhanca[mascara_spike])
        
    return espaco_k_corrigido









# FUNÇÕES PARA PLOTAR OS GRÁFICOS DAS QUESTÕES 

def plotar_fatias_q1(raw_data, fatias):
    # Plota todas as fatias da Questão 1 juntas em uma única figura (2 linhas x N colunas).
    # Linha 1: Espaço-k (Magnitude Log)
    # Linha 2: Imagem Reconstruída
    
    num_fatias = len(fatias)
    fig, axes = plt.subplots(2, num_fatias, figsize=(18, 7))
    
    for i, fatia_1based in enumerate(fatias):
        idx_fatia = fatia_1based - 1
        kspace_fatia = raw_data[:, :, idx_fatia]
        img_rec = reconstruir_imagem_2d(kspace_fatia)
        
        # Linha superior: Espaço-k
        mag_k_log = np.log(1 + np.abs(kspace_fatia))
        axes[0, i].imshow(mag_k_log, cmap='gray')
        axes[0, i].set_title(f"Espaço-k (Fat #{fatia_1based})")
        axes[0, i].axis('off')
        
        # Linha inferior: Imagem Reconstruída
        axes[1, i].imshow(img_rec, cmap='gray')
        axes[1, i].set_title(f"Imagem (Fat #{fatia_1based})")
        axes[1, i].axis('off')
        
    plt.suptitle("Questão 1 - Reconstrução das Fatias Selecionadas", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plotar_comparativo_subamostragem(img_orig, k_orig, img_r2, k_r2, img_r4, k_r4):
    #Plota o comparativo das reconstruções da Questão 2 (2 linhas x 3 colunas).
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Linha superior: Espaços-k
    axes[0, 0].imshow(np.log(1 + np.abs(k_orig)), cmap='gray')
    axes[0, 0].set_title("Espaço-k Original (Fat #13)")
    axes[0, 1].imshow(np.log(1 + np.abs(k_r2)), cmap='gray')
    axes[0, 1].set_title("Espaço-k Sub-amostrado (R=2)")
    axes[0, 2].imshow(np.log(1 + np.abs(k_r4)), cmap='gray')
    axes[0, 2].set_title("Espaço-k Sub-amostrado (R=4)")
    
    # Linha inferior: Imagens Reconstruídas
    axes[1, 0].imshow(img_orig, cmap='gray')
    axes[1, 0].set_title("Imagem Original")
    axes[1, 1].imshow(img_r2, cmap='gray')
    axes[1, 1].set_title("Reconstrução (Aliasing R=2)")
    axes[1, 2].imshow(img_r4, cmap='gray')
    axes[1, 2].set_title("Reconstrução (Aliasing R=4)")
    
    for ax in axes.ravel():
        ax.axis('off')
        
    plt.suptitle("Questão 2 - Efeito do Sub-amostramento (Aliasing)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plotar_correcao_spike(kspace_spike, img_spike, kspace_limpo, img_limpa):
    # Plota um painel 2x2 comparando o Espaço-k e a Imagem Reconstruída antes e depois da remoção do Spike (Herringbone).

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Linha 1: Com Spike
    mag_k_spike = np.log(1 + np.abs(kspace_spike))
    axes[0, 0].imshow(mag_k_spike, cmap='gray')
    axes[0, 0].set_title("Espaço-k (Com Spike)")
    
    axes[0, 1].imshow(img_spike, cmap='gray')
    axes[0, 1].set_title("Imagem Reconstruída (Com Herringbone)")
    
    # Linha 2: Spike Removido
    mag_k_limpo = np.log(1 + np.abs(kspace_limpo))
    axes[1, 0].imshow(mag_k_limpo, cmap='gray')
    axes[1, 0].set_title("Espaço-k (Spike Removido)")
    
    axes[1, 1].imshow(img_limpa, cmap='gray')
    axes[1, 1].set_title("Imagem Reconstruída (Corrigida)")
    
    for ax in axes.ravel():
        ax.axis('off')
        
    plt.suptitle("Questão 3 - Detecção e Correção do Artefato Spike", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()





# EXECUÇÃO PRINCIPAL

if __name__ == '__main__':
    #Só um aviso pra quem não souber: os arquivos rawdata.mat e spike.mat tem que estar na mesma pasta do código, se não tem que alterar o caminho no carregar_dados_mat

    params = definir_parametros_estudo()


    
    # QUESTÃO 1: Reconstrução conjunta das fatias 6, 10, 14, 18 e 22
    print("--- Processando Questão 1 ---")
    raw_data = carregar_dados_mat('rawdata.mat', 'rawdata')
    plotar_fatias_q1(raw_data, params['fatias_q1'])



    # QUESTÃO 2: Sub-amostragem na Fatia #13 (R=2 e R=4)
    print("--- Processando Questão 2 ---")
    fatia_13_idx = params['fatia_q2'] - 1
    kspace_fatia13 = raw_data[:, :, fatia_13_idx]
    
    img_orig = reconstruir_imagem_2d(kspace_fatia13)
    
    kspace_r2 = subamostrar_espaco_k(kspace_fatia13, fator_r=2)
    img_r2 = reconstruir_imagem_2d(kspace_r2)
    
    kspace_r4 = subamostrar_espaco_k(kspace_fatia13, fator_r=4)
    img_r4 = reconstruir_imagem_2d(kspace_r4)
    
    plotar_comparativo_subamostragem(img_orig, kspace_fatia13, img_r2, kspace_r2, img_r4, kspace_r4)
    



    # QUESTÃO 3: Remoção do Spike (Herringbone)
    print("--- Processando Questão 3 ---")
    spike_data = carregar_dados_mat('spike.mat', 'spike')
    kspace_spike = spike_data[:, :, 0] if spike_data.ndim == 3 else spike_data

    img_spike = reconstruir_imagem_2d(kspace_spike)
    
    kspace_limpo = remover_spike_espaco_k(kspace_spike, limiar_std=params['limiar_spike_std'])
    img_limpa = reconstruir_imagem_2d(kspace_limpo)
    
    plotar_correcao_spike(kspace_spike, img_spike, kspace_limpo, img_limpa)