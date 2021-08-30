# Tutorial


## Instruções de instalação

- [libraries_instructions.txt](libraries_instructions.txt): Apresenta a linguagem e as bibliotecas utilizadas nos arquivos deste trabalho.


## Comandos para executar os arquivos

- [commands_CLI.txt](commands_CLI.txt): Contém os comandos de execução dos arquivos utilizados neste trabalho.


## Arquivos

### Conjunto de arquivos para realizar o pré-processamento das bases de dados

- [preprocess_datasets.py](Code/preprocess_datasets/preprocess_datasets.py): Responsável por construir a base de dados deste trabalho.
- [How_to_Run.txt](Code/preprocess_datasets/How_to_Run.txt): Tutorial de como realizar o pré-processamento das bases de dados.


### Conjunto de arquivos para realizar o pré-processamento das imagens

- [preprocess_all_images.py](Code/preprocess_images/preprocess_all_images.py): Realiza o pré-processamento de várias imagens.
- [preprocess_one_image.py](Code/preprocess_images/preprocess_one_image.py): Realiza o pré-processamento de apenas uma imagem (utilizado para testes).


### Conjunto de arquivos principal

- [train.py](Code/train.py): Responsável por treinar o modelo.
- [test.py](Code/test.py): Responsável por testar o modelo.
- [metrics.py](Code/metrics.py): Responsável por criar a matriz de confusão e calcular suas respectivas métricas.


### Conjunto de arquivos para construção dos ensembles

- [get_best_ensemble_combination.py](Code/ensembles/get_best_ensemble_combination.py): Responsável por encontrar a melhor combinação de modelos para a construção ensembles.
- [ensemble_vote.py](Code/ensembles/ensemble_vote/ensemble_vote.py): Responsável por construir ensembles com a estratégia de sumarização chamada de "MaxVotos".
- [ensemble_average.py](Code/ensembles/ensemble_average/ensemble_average.py): Responsável por construir ensembles com a estratégia de sumarização chamada de "Média".
- [ensemble_maximum.py](Code/ensembles/ensemble_maximum/ensemble_maximum.py): Responsável por construir ensembles com a estratégia de sumarização chamada de "MaxProb".


### Conjunto de arquivos para aplicação do método dos gradientes integrados nas imagens

- [get_heatmaps_all_images.py](Code/heatmaps/get_heatmaps/get_heatmaps_all_images.py): Aplica o método dos gradientes integrados em várias imagens.
- [get_heatmaps_one_image.py](Code/heatmaps/get_heatmaps/get_heatmaps_one_image.py): Aplica o método dos gradientes integrados em apenas uma imagem (utilizado para testes).
- [emphasize_heatmap_intensity_all_images.py](Code/heatmaps/emphasize_heatmap_intensity/emphasize_heatmap_intensity_all_images.py): Enfatiza a cor dos gradientes integrados em várias imagens.
- [emphasize_heatmap_intensity_one_image.py](Code/heatmaps/emphasize_heatmap_intensity/emphasize_heatmap_intensity_one_image.py): Enfatiza a cor dos gradientes integrados em apenas uma imagem (utilizado para testes).


### Conjunto de arquivos relacionados a informações ou imagens deste trabalho

- [paper_images](paper_images): Contém as imagens utilizadas na confecção deste trabalho.
- [tests_results](tests_results): Contém o resultado dos testes de cada um dos modelos, as matrizes de confusão e as métricas referentes às matrizes.
- [trained_models](trained_models): Contém as informações de treinamento e validação de cada um dos modelos (devido ao tamanho dos modelos não foi possível colocá-los aqui no github).
