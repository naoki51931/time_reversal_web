import React, { useState } from "react";

const GenerateForm = () => {
  const API_URL = "http://13.158.23.179:8000/generate";
  const [file1, setFile1] = useState(null);
  const [file2, setFile2] = useState(null);
  const [frames, setFrames] = useState(8);
  const [t0, setT0] = useState(5);
  const [sChurn, setSChurn] = useState(0.5);
  const [noNoise, setNoNoise] = useState(false);
  const [loading, setLoading] = useState(false);
  const [downloadUrl, setDownloadUrl] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file1 || !file2) {
      alert("2枚の画像をアップロードしてください");
      return;
    }

    setLoading(true);
    setDownloadUrl(null);

    const formData = new FormData();
    formData.append("file1", file1);
    formData.append("file2", file2);
    formData.append("frames", frames);
    formData.append("t0", t0);
    formData.append("s_churn", sChurn);
    formData.append("w_o_noise_re_injection", noNoise);
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5 * 60 * 1000); // 5分

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
	signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error("生成に失敗しました");
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      setDownloadUrl(url);
    } catch (err) {
      alert(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: "600px", margin: "0 auto" }}>
      <h2>🎬 Time-Reversal 動画生成</h2>
      <form onSubmit={handleSubmit}>
        <div>
          <label>画像1:</label>
          <input type="file" accept="image/*" onChange={(e) => setFile1(e.target.files[0])} />
        </div>
        <div>
          <label>画像2:</label>
          <input type="file" accept="image/*" onChange={(e) => setFile2(e.target.files[0])} />
        </div>

        <div>
          <label>フレーム数 (M):</label>
          <input type="number" value={frames} onChange={(e) => setFrames(e.target.value)} />
        </div>
        <div>
          <label>t0 (cutoff timestep):</label>
          <input type="number" value={t0} onChange={(e) => setT0(e.target.value)} />
        </div>
        <div>
          <label>s_churn:</label>
          <input
            type="number"
            step="0.1"
            value={sChurn}
            onChange={(e) => setSChurn(e.target.value)}
          />
        </div>
        <div>
          <label>
            <input
              type="checkbox"
              checked={noNoise}
              onChange={(e) => setNoNoise(e.target.checked)}
            />
            noise再注入なし
          </label>
        </div>

        <button type="submit" disabled={loading}>
          {loading ? "生成中..." : "生成する"}
        </button>
      </form>

      {downloadUrl && (
        <div style={{ marginTop: "20px" }}>
          <a href={downloadUrl} download="result.zip">
            ⬇️ ZIPファイルをダウンロード
          </a>
        </div>
      )}
    </div>
  );
};

export default GenerateForm;
