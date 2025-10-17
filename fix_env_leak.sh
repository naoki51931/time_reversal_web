#!/usr/bin/env bash
set -e

echo "🧹 === GitHub Push Protection 修復スクリプト ==="
echo "プロジェクトルートで実行してください。"

# 仮想環境チェック
if [ -d "backend/venv" ]; then
  echo "✅ venv 検出。仮想環境を有効化します..."
  source backend/venv/bin/activate
else
  echo "⚠️ 仮想環境が見つかりません。スキップします。"
fi

# git-filter-repo インストール
if ! command -v git-filter-repo &> /dev/null; then
  echo "📦 git-filter-repo が見つかりません。pipxでインストールします..."
  sudo apt update -y
  sudo apt install -y pipx
  pipx install git-filter-repo
else
  echo "✅ git-filter-repo は既にインストール済みです。"
fi

# .env.swp を削除
echo "🗑️ 履歴から .env.swp を完全削除中..."
git filter-repo --path backend/.env.swp --invert-paths || true

# .gitignore 強化
echo ".env" >> .gitignore
echo ".env.swp" >> .gitignore
git add .gitignore
git commit -m "Add .env and .env.swp to .gitignore" || true

# 強制 push
echo "🚀 main ブランチを GitHub に強制 push します..."
git push origin main --force

echo "✅ 完了しました！"
echo "🔐 次にやること:"
echo "1️⃣ Hugging Face の古いトークンを削除"
echo "2️⃣ 新しいトークンを発行して backend/.env に保存"
echo "3️⃣ push 前に .env がコミットされていないことを確認"

echo "✨ Push Protection によるブロックが解消されているはずです！"

