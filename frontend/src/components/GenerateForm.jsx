import React, { useState } from "react";

function GenerateForm() {
  const [file1, setFile1] = useState(null);
  const [file2, setFile2] = useState(null);
  const [loading, setLoading] = useState(false);
  const [videoUrl, setVideoUrl] = useState(null);
  const [zipUrl, setZipUrl] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file1 || !file2) return;

    setLoading(true);
    setVideoUrl(null);
    setZipUrl(null);

    const formData = new FormData();
    formData.append("file1", file1);
    formData.append("file2", file2);
    formData.append("frames", 16); // 中間フレーム数

    try {
      const res = await fetch("http://localhost:8000/generate", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error("生成に失敗しました");
      }

      const blob = await res.blob();
      const zipObjectUrl = URL.createObjectURL(blob);
      setZipUrl(zipObjectUrl);

      // --- ZIP の中から mp4 を抽出してプレビューする場合 ---
      // jszip を利用
      const JSZip = (await import("jszip")).default;
      const zip = await JSZip.loadAsync(blob);
      const mp4File = zip.file("result.mp4");
      if (mp4File) {
        const mp4Blob = await mp4File.async("blob");
        setVideoUrl(URL.createObjectURL(mp4Blob));
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: "20px" }}>
      <h2>Time-Reversal アニメ生成</h2>
      <form onSubmit={handleSubmit}>
        <div>
          <input type="file" accept="image/*" onChange={(e) => setFile1(e.target.files[0])} />
        </div>
        <div>
          <input type="file" accept="image/*" onChange={(e) => setFile2(e.target.files[0])} />
        </div>
        <button type="submit" disabled={loading}>
          {loading ? "生成中..." : "生成する"}
        </button>
      </form>

      {zipUrl && (
        <div style={{ marginTop: "20px" }}>
          <a href={zipUrl} download="result.zip">ZIPをダウンロード</a>
        </div>
      )}

      {videoUrl && (
        <div style={{ marginTop: "20px" }}>
          <video controls src={videoUrl} style={{ width: "480px" }} />
        </div>
      )}
    </div>
  );
}

export default GenerateForm;
