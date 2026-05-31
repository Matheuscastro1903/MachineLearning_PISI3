# 1. Utiliza a versão leve do Python 3.11 (estável)
FROM python:3.11-slim

# 2. Define o diretório de trabalho dentro do servidor
WORKDIR /app

# 3. Copia todo o conteúdo da sua pasta "dash" para dentro do servidor
COPY dash/ /app/

# 4. Instala os pacotes necessários (como o requirements.txt agora está dentro da pasta dash, ele vai encontrá-lo)
RUN pip install --no-cache-dir -r requirements.txt

# 5. REGRA DO HUGGING FACE: A aplicação TEM de escutar na porta 7860
ENV PORT=7860
EXPOSE 7860

# 6. Comando final para iniciar a aplicação
CMD ["python", "app.py"]