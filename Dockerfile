# 使用官方的 Python 3.9 映像檔作為基礎
FROM python:3.9-slim

# 安裝 whisper 需要的系統工具 ffmpeg
RUN apt-get update && apt-get install -y ffmpeg

# 設定工作目錄
WORKDIR /app

# 複製 requirements.txt 並安裝依賴套件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製所有專案檔案到工作目錄
COPY . .

# 開放容器的 10000 連接埠
EXPOSE 10000

# 容器啟動時要執行的指令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]