🌀 Time Reversal Web — AI中間フレーム生成アプリ

2枚の画像（原画A → 原画B）の間を補間して、
AIによる中間フレームを自動生成するWebアプリケーション。
アニメーションの中割り・線画補完・モーション補間・VAE潜在補間を統合的に扱えるツールです。

🌐 全体構成
層	技術	内容
Frontend	React + Axios + Material-UI	アップロード・パラメータ設定・生成結果プレビュー
Backend	FastAPI + Diffusers + OpenCV + Pillow	Stable Diffusion VAE補間・線画抽出・ノイズ除去・動作補間
モデル	Stable Diffusion 1.5 (runwayml/stable-diffusion-v1-5)	Diffusersを用いて潜在空間補間を実行
🧭 機能一覧
機能	内容
🔹 通常補間モード	VAE潜在空間での線形補間による中間フレーム生成
🔹 線画抽出モード	Canny + CLAHE + シャープ化で線画抽出
🔹 ノイズ除去モード	fastNlMeansDenoisingColored によるノイズ低減
🔹 線画＋ノイズ除去ハイブリッドモード	コントラスト強調・γ補正・線膨張処理を組み合わせた高精度描線補完
🔹 時反転サンプリングモード（TRS）	Stable Diffusion を用いた Time Reversal Sampling により、潜在空間上で自然な時系列遷移を補間
🔹 動作補間モード（Motion）	顔・体・姿勢の変化を自動検出し、動きの中間状態を複数フレーム生成
🔹 フレーム数指定生成	frames に応じて自動的に中間コマを生成（例：8 → 8枚）
🔹 プレビュー機能	出力結果をサムネイル＋拡大で確認可能
🔹 API統合生成	/generate エンドポイントで一括処理が可能
🧩 ディレクトリ構成
time_reversal_web/
├── backend/
│   ├── main.py                      # FastAPI メインサーバー
│   └── models/
│       ├── pipeline_time_reversal.py
│       ├── pipeline_time_reversal_lineart.py
│       ├── pipeline_time_reversal_denoise.py
│       ├── pipeline_time_reversal_sampling.py  # ← 時反転サンプリング(TRS)
│       └── pipeline_motion_auto.py              # ← 動作補間モード
└── frontend/
    └── src/
        ├── App.jsx
        ├── GenerateForm.jsx                    # React UI（チェックボックスで各モード切替）
        └── index.js

⚙️ バックエンド構成
main.py（概要）

FastAPI による生成API。
画像2枚と補間パラメータを受け取り、指定モードでフレーム群を生成する。

モード一覧
モード名	処理内容
normal	VAE潜在補間（ベースライン）
lineart	線画抽出処理（Canny + CLAHE + UnsharpMask）
denoise	ノイズ除去（fastNlMeansDenoisingColored）
lineart_denoise	コントラスト×2.4 + 線膨張 + γ補正
diffusion_trs	Stable Diffusion 時反転サンプリング（潜在空間のTRS補間）
motion	顔や体の自然な動きをAIが推定し、複数中間フレームを生成
時反転サンプリング（TRS）

Diffusers の StableDiffusionPipeline を利用

潜在ベクトル z(t) を時間反転（t→−t）で補間

tau_step により補間粒度を制御

Guidance Scale による生成強度調整

動作補間モード（Motion）

A/Bの画素差分から変化量を推定

差分に応じて「顔」「上半身」「全身」などの動作プロンプトを自動生成

frames 数に応じて線形ブレンドした画像をStable Diffusionに通し自然補正

生成例：

motion_frame_00.png
motion_frame_01.png
motion_frame_02.png
...

💻 フロントエンド構成
GenerateForm.jsx（React UI）
主なUI機能

画像A / 画像B のアップロード

frames（生成フレーム数）と t0（時間パラメータ）の入力

チェックボックスで各モード選択：

線画抽出

ノイズ除去

線画＋ノイズ除去ハイブリッド

Stable Diffusion 時反転補間（TRS）

動作補間モード（Motion）

結果をプレビュー・スライド切替（← → キー対応）

API仕様
POST /generate
Content-Type: multipart/form-data
Body:
  image_1: file
  image_2: file
  frames: int
  t0: float
  lineart: bool
  denoise: bool
  diffusion_trs: bool
  motion: bool


レスポンス例：

{
  "status": "ok",
  "frames_generated": 5,
  "image_urls": [
    "/outputs/motion_frame_00.png",
    "/outputs/motion_frame_01.png",
    "/outputs/motion_frame_02.png"
  ],
  "debug": { "mode": "motion_auto" }
}

🧠 技術詳細
カテゴリ	使用技術
モデル	diffusers, torch
画像処理	Pillow, OpenCV
サーバー	FastAPI, Uvicorn
Web UI	React, Axios, Material-UI
データ形式	FormData, JSON
精度設定	fp32固定（Half混在エラー防止）
🧩 処理パイプライン概要
Input A, B
   │
   ├─→ Lineart mode → エッジ抽出・線強調
   ├─→ Denoise mode → ノイズ低減
   ├─→ TRS mode → 潜在空間で時反転補間
   └─→ Motion mode → 自動動作プロンプトで中間動作生成
         └─→ frames数に応じて自然な連続動作を出力

🧪 実行例
サーバー起動
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

React起動
cd ../frontend
npm start

例1: 線画＋ノイズ除去
入力	出力例
A（輪郭） → B（完成）	/outputs/frame_003_strongline.png
効果: コントラスト強化＆線残しで中間線が鮮明	
例2: 動作補間モード
入力	出力例
A（目を開けた顔） → B（目を閉じた顔）	/outputs/motion_frame_00.png〜motion_frame_07.png
効果: 自動プロンプトで「まばたき動作」を補間し連続フレームを生成	
例3: 時反転補間モード
入力	出力例
A（昼景） → B（夕景）	/outputs/trs_frame_00.png〜trs_frame_07.png
効果: Stable Diffusionにより照明・空気感の自然な中間補間を実現	
🚧 今後の開発予定

 Retinexベースのトーン補正（明暗均質化）

 出力連番フレームの自動GIF生成

 ReactプレビューでGIF再生

 CORS設定の安定化（外部アクセス強化）

 パラメータプリセット（線画・柔和補間・強シャープ補間）

 ControlNet連携によるポーズ条件付き補間

🪄 出力モード比較
モード	処理内容	出力例
Normal	標準VAE補間	frame_003.png
Lineart	線画抽出	frame_003_line.png
Denoise	ノイズ除去	frame_003_denoised.png
Hybrid	線＋ノイズ除去強化	frame_003_strongline.png
TRS	時反転サンプリング補間	trs_frame_003.png
Motion	動作自動補間	motion_frame_003.png
📄 ライセンス

本アプリは研究・教育・クリエイティブ用途での使用を想定しています。
Stable Diffusion / Diffusers のライセンス条件に従ってください。

👤 開発者情報

Naoki Ueda
Backend / Frontend Developer
Generative AI × FastAPI × Diffusers
GitHub: https://github.com/naoki51931/time_reversal_web

🧩 ChatGPT共同開発ノート

このアプリは ChatGPT と共同設計されたAI補間システムです。
バックエンド（FastAPI + Diffusers）および React UI はすべて対話を通じて構築されています。
開発引継ぎは docs/Handoff_Script.md を参照。

📘 最終更新日: 2025-10-14
📦 対応モード: Normal / Lineart / Denoise / Hybrid / TRS / Motion
🎨 目的: アニメ中割り・線画補完・ポーズ補間・自然動作生成

"AIが描く、時間と動きの中間点。"
