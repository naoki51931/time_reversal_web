import React, { useState } from "react";
import axios from "axios";

const API_BASE_URL = "http://43.207.92.186:8000"; // FastAPIのURLに合わせて修正

export default function GenerateForm() {
  const [imageA, setImageA] = useState(null);
  const [imageB, setImageB] = useState(null);
  const [frames, setFrames] = useState(3);
  const [t0, setT0] = useState(5.0);
  const [lineart, setLineart] = useState(false);
  const [resultUrls, setResultUrls] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!imageA || !imageB) {
      alert("画像を2枚選択してください");
      return;
    }
    setLoading(true);
    setResultUrls([]);

    const formData = new FormData();
    formData.append("image_1", imageA);
    formData.append("image_2", imageB);
    formData.append("frames", frames);
    formData.append("t0", t0);
    formData.append("lineart", lineart); // ✅ チェックボックスの値を送信

    try {
      console.log("[Client] Sending request to:", `${API_BASE_URL}/generate`);
      const res = await axios.post(`${API_BASE_URL}/generate`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      console.log("[Client] Response:", res.data);
      setResultUrls(res.data.image_urls || []);
    } catch (err) {
      console.error("Error during generation:", err);
      alert("生成中にエラーが発生しました");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: "1rem", fontFamily: "sans-serif" }}>
      <h2>中割生成フォーム</h2>
      <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: "0.8rem" }}>
        <label>
          画像A:
          <input type="file" accept="image/*" onChange={(e) => setImageA(e.target.files[0])} />
        </label>
        <label>
          画像B:
          <input type="file" accept="image/*" onChange={(e) => setImageB(e.target.files[0])} />
        </label>
        <label>
          フレーム数:
          <input
            type="number"
            value={frames}
            min="2"
            max="20"
            onChange={(e) => setFrames(e.target.value)}
          />
        </label>
        <label>
          t0（補間の滑らかさ）:
          <input
            type="number"
            step="0.1"
            value={t0}
            onChange={(e) => setT0(e.target.value)}
          />
        </label>
        <label>
          <input
            type="checkbox"
            checked={lineart}
            onChange={(e) => setLineart(e.target.checked)}
          />
          線画抽出モードで出力する
        </label>
        <button type="submit" disabled={loading}>
          {loading ? "生成中..." : "生成開始"}
        </button>
      </form>

      {resultUrls.length > 0 && (
        <div style={{ marginTop: "1.5rem" }}>
          <h3>生成結果</h3>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))",
              gap: "0.5rem",
            }}
          >
            {resultUrls.map((url, i) => (
              <img
                key={i}
                src={url}
                alt={`frame_${i}`}
                style={{
                  width: "100%",
                  borderRadius: "8px",
                  boxShadow: "0 0 4px rgba(0,0,0,0.2)",
                }}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

