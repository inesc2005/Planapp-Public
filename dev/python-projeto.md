# Setup do Ambiente de Desenvolvimento Python

Este documento descreve como configurar o ambiente de desenvolvimento para este projeto, utilizando `uv` para uma gestão de dependências rápida e eficiente.

## Pré-requisitos

Antes de começar, garanta que tem o `uv` instalado no seu sistema. `uv` é um instalador e gestor de dependências Python extremamente rápido, desenvolvido pela Astral.

Se não tiver o `uv`, pode instalá-lo seguindo as instruções no [site oficial da Astral](https://astral.sh/uv).

**Windows (PowerShell):**
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

**macOS/Linux (Shell):**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Passos para Configuração

1.  **Clonar o Repositório (se ainda não o fez):**
    ```bash
    git clone <URL_DO_REPOSITORIO>
    cd <NOME_DO_PROJETO>
    ```

2.  **Criar o Ambiente Virtual:**
    Use `uv` para criar um ambiente virtual. Por convenção, este ambiente é criado numa pasta chamada `.venv` na raiz do projeto.
    ```bash
    uv venv
    ```

3.  **Instalar as Dependências (Libs/Packages):**
    Com o ambiente virtual criado, o próximo passo é instalar todas as dependências (bibliotecas/pacotes) listadas no ficheiro `pyproject.toml`. O comando `uv sync` garante que o seu ambiente tem exatamente as mesmas versões de pacotes especificadas no ficheiro de lock (`uv.lock`), garantindo reprodutibilidade.
    ```bash
    uv sync
    ```

4.  **Configurar Variáveis de Ambiente:**
    Crie um ficheiro chamado `.env` na raiz do projeto. Este ficheiro é utilizado para guardar segredos e configurações específicas do ambiente, como chaves de API.

    O conteúdo do ficheiro será enviado por e-mail por questões de segurança. Cole o conteúdo enviado no ficheiro `.env` que acabou de criar.

    **Exemplo da estrutura do ficheiro `.env`:**
    ```properties
    SUPABASE_URL=...
    SUPABASE_ANON_KEY=...
    SUPABASE_TABLE=...
    ```

5.  **Ativar o Ambiente Virtual:**
    Para usar as ferramentas e bibliotecas instaladas, precisa de ativar o ambiente.

    **Windows (PowerShell):**
    ```powershell
    .venv\Scripts\Activate.ps1
    ```

    **macOS/Linux (Shell):**
    ```bash
    source .venv/bin/activate
    ```

Após estes passos, o seu terminal estará a usar o interpretador Python e as bibliotecas do ambiente `.venv`, e estará pronto para executar os scripts do projeto, como `src/ingest.py`.
