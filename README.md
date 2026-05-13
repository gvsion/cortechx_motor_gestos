# Motor de gestos (visão computacional)

Pipeline em Python que usa a **câmera** e **MediaPipe Hands** para mover o cursor na tela, reconhecer **pinças** (clique esquerdo, direito e arraste) e **rolagem** com gesto de dois dedos, pensada para **totens** ou quiosques com Linux.

## O que inclui

- **Rastreamento**: landmarks das mãos em tempo real (`src/hand_tracker.py`).
- **Mapeamento**: ponta do indicador → coordenadas da tela, com margem configurável e suavização EMA (`src/mapping.py`).
- **Gestos**: pinça índice–polegar (clique / arraste), pinça médio–polegar (clique direito) com histerese e tempos de toque (`src/gesture_interactor.py`).
- **Scroll**: modo com “trava” — entra com pose mais rígida durante alguns frames; com o modo ativo usa pose relaxada para não cortar o gesto (`src/scroll_control.py`, `src/gesture_math.py`).
- **Pipeline unificada** (`src/pipeline.py`): um `GestureMotor` por frame devolve cursor, eventos, scroll e estado de debug.
- **Debug visual** (`run_debug.py` + `src/debug_overlay.py`): câmera, HUD e retângulo de mapeamento; opção de ecrã inteiro ou janela.

## Requisitos

- **Python** 3.10 ou superior (recomendado 3.12).
- **Câmera** USB ou integrada.
- **Linux**: para `--inject-mouse`, o pacote `pynput` controla o cursor do sistema; em alguns ambientes pode ser preciso permissão de acessibilidade ou execução fora de sandbox restritiva.

## Instalação

```bash
cd motor_gestos_vc
python3 -m venv .venv
source .venv/bin/activate   # no Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Dependências principais: `opencv-python`, `mediapipe`, `numpy`, `pynput`.

## Como executar

### Modo debug (com janela OpenCV)

```bash
python run_debug.py
```

Por defeito tenta **ecrã inteiro** e escala o vídeo ao tamanho do monitor (letterbox). Para janela redimensionável:

```bash
python run_debug.py --windowed
```

### Modo totem (sem janela)

```bash
python run_totem.py
```

Equivale a `run_debug.py --headless` (útil em máquina sem display ou serviço em segundo plano). Pare com `Ctrl+C`.

### Injetar mouse no sistema

```bash
python run_debug.py --inject-mouse
# ou
python run_totem.py --inject-mouse
```

### Alguns parâmetros úteis

| Opção | Descrição |
|--------|-----------|
| `--camera N` | Índice da câmera (padrão `0`). |
| `--no-mirror` | Desativa espelhamento horizontal do vídeo. |
| `--screen-width` / `--screen-height` | Resolução lógica do totem; se omitido, tenta detetar o ecrã. |
| `--margin` | Margem normalizada do retângulo de mapeamento (padrão `0.12`; menor = área ativa maior no vídeo). |
| `--ema` | Suavização do cursor (0–1). |
| `--max-step` | Limite de pixels por frame no cursor (anti-salto); `0` desliga. |
| `--no-scroll` | Desativa rolagem por gesto. |
| `--scroll-invert` / `--scroll-sensitivity` | Sentido e ganho do scroll. |
| `--headless` | Sem OpenCV (já implícito em `run_totem.py`). |
| `--log-events` | Com `--headless`, imprime eventos de gesto no stderr. |

## Estrutura do código

```
motor_gestos_vc/
├── run_debug.py      # Entrada principal: debug + mesmas opções do totem
├── run_totem.py      # Atalho para modo headless
├── requirements.txt
├── src/
│   ├── pipeline.py       # GestureMotor, GestureMotorConfig, MotorOutput
│   ├── hand_tracker.py   # MediaPipe + conversão de frame
│   ├── capture.py        # Abstração da câmera (OpenCV)
│   ├── mapping.py        # HandToScreenMapper, deteção de tamanho do ecrã
│   ├── gesture_interactor.py
│   ├── gesture_math.py   # Razões e poses (pinch, scroll)
│   ├── scroll_control.py
│   ├── debug_overlay.py
│   └── mouse_inject.py   # Injeção via pynput (opcional)
└── .vscode/launch.json   # Perfis de debug (ecrã inteiro / janela / totem)
```

## Desenvolvimento no VS Code / Cursor

Abra a pasta do projeto e use **Run and Debug** com um dos perfis em `.vscode/launch.json` (por exemplo *Motor de Gestos: run_debug (ecrã inteiro)*).

## Licença

Indique aqui a licença do repositório (por exemplo MIT ou GPL-3.0) quando a definir.
