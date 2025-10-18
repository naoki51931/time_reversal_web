import React, { useState, useCallback, useEffect } from "react";
import axios from "axios";
import {
  Button,
  TextField,
  Typography,
  Box,
  LinearProgress,
  Checkbox,
  FormControlLabel,
  Dialog,
  DialogContent,
  DialogTitle,
  IconButton,
} from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
import ArrowBackIosNewIcon from "@mui/icons-material/ArrowBackIosNew";
import ArrowForwardIosIcon from "@mui/icons-material/ArrowForwardIos";

export default function GenerateForm() {
  const [image1, setImage1] = useState(null);
  const [image2, setImage2] = useState(null);
  const [frames, setFrames] = useState(8);
  const [t0, setT0] = useState(0.0);
  const [lineart, setLineart] = useState(false);
  const [denoise, setDenoise] = useState(false);
  const [diffusionTrs, setDiffusionTrs] = useState(false);
  const [motion, setMotion] = useState(false);

  // ✅ 初期値（環境指定）
  const [strength, setStrength] = useState(0.15);
  const [guidance, setGuidance] = useState(20.0);

  const [imageUrls, setImageUrls] = useState([]);
  const [loading, setLoading] = useState(false);
  const [openDialog, setOpenDialog] = useState(false);
  const [currentIndex, setCurrentIndex] = useState(0);

  const API_BASE = "http://13.159.71.138:8000";

  // === 矢印キー移動 ===
  const handleNext = useCallback(() => {
    if (imageUrls.length === 0) return;
    setCurrentIndex((prev) => (prev + 1) % imageUrls.length);
  }, [imageUrls]);

  const handlePrev = useCallback(() => {
    if (imageUrls.length === 0) return;
    setCurrentIndex((prev) => (prev - 1 + imageUrls.length) % imageUrls.length);
  }, [imageUrls]);

  // === キーボード操作対応 ===
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (!openDialog || imageUrls.length === 0) return;
      if (e.key === "ArrowLeft") handlePrev();
      if (e.key === "ArrowRight") handleNext();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [openDialog, imageUrls, handleNext, handlePrev]);

  // === 画像プリロード ===
  const preloadImages = useCallback((urls) => {
    urls.forEach((url) => {
      const img = new Image();
      img.src = url.startsWith("http")
        ? url
        : `${API_BASE}${url.startsWith("/") ? url : "/" + url}`;
    });
  }, []);

  // === 送信 ===
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!image1 || !image2) {
      alert("2枚の画像を選択してください。");
      return;
    }

    const formData = new FormData();
    formData.append("image_1", image1);
    formData.append("image_2", image2);
    formData.append("frames", frames);
    formData.append("t0", t0);
    formData.append("lineart", lineart);
    formData.append("denoise", denoise);
    formData.append("diffusion_trs", diffusionTrs);
    formData.append("motion", motion);
    formData.append("strength", strength);
    formData.append("guidance", guidance);

    try {
      setLoading(true);

      const res = await axios.post(`${API_BASE}/generate`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      console.log("📦 FastAPIレスポンス全体:", res.data);
      console.log("🖼️ image_urls:", res.data.image_urls);

      if (res.data.image_urls && res.data.image_urls.length > 0) {
        // ✅ 既存の表示を一旦クリアして確実に再レンダーさせる
        setImageUrls([]);
        setCurrentIndex(0);

        setTimeout(() => {
          const newUrls = [...res.data.image_urls]; // 新しい配列参照を強制
          preloadImages(newUrls);
          setImageUrls(newUrls);
        }, 10);
      } else {
        alert("生成結果がありません。");
      }
    } catch (err) {
      console.error("❌ エラー発生:", err);
      alert("生成中にエラーが発生しました。");
    } finally {
      setLoading(false);
    }
  };

  const handleImageClick = (index) => {
    setCurrentIndex(index);
    setOpenDialog(true);
  };

  // === URLを安全に処理 ===
  const resolveUrl = (url) => {
    return url.startsWith("http")
      ? url
      : `${API_BASE}${url.startsWith("/") ? url : "/" + url}`;
  };

  // === 新しい画像セット時は自動的に最初の画像を表示 ===
  useEffect(() => {
    if (imageUrls.length > 0) {
      setCurrentIndex(0);
    }
  }, [imageUrls]);

  return (
    <Box sx={{ width: "530px", margin: "0 auto", mt: 4, textAlign: "center" }}>
      <Typography variant="h4" gutterBottom>
        Time Reversal Generator
      </Typography>

      <form onSubmit={handleSubmit}>
        <Box sx={{ my: 2 }}>
          <Typography>画像A</Typography>
          <input
            type="file"
            accept="image/*"
            onChange={(e) => setImage1(e.target.files[0])}
          />
        </Box>
        <Box sx={{ my: 2 }}>
          <Typography>画像B</Typography>
          <input
            type="file"
            accept="image/*"
            onChange={(e) => setImage2(e.target.files[0])}
          />
        </Box>

        <TextField
          label="生成フレーム数"
          type="number"
          value={frames}
          onChange={(e) => setFrames(e.target.value)}
          fullWidth
          sx={{ my: 1 }}
        />
        <TextField
          label="t0 時間パラメータ"
          type="number"
          value={t0}
          onChange={(e) => setT0(e.target.value)}
          fullWidth
          sx={{ my: 1 }}
        />

        <FormControlLabel
          control={
            <Checkbox
              checked={lineart}
              onChange={(e) => setLineart(e.target.checked)}
            />
          }
          label="線画抽出モード"
        />
        <FormControlLabel
          control={
            <Checkbox
              checked={denoise}
              onChange={(e) => setDenoise(e.target.checked)}
            />
          }
          label="ノイズ除去モード"
        />
        <FormControlLabel
          control={
            <Checkbox
              checked={diffusionTrs}
              onChange={(e) => setDiffusionTrs(e.target.checked)}
            />
          }
          label="Stable Diffusion 時反転補間モード"
        />
        <FormControlLabel
          control={
            <Checkbox
              checked={motion}
              onChange={(e) => setMotion(e.target.checked)}
            />
          }
          label="動作補間モード"
        />

        <TextField
          label="強度 (strength)"
          type="number"
          value={strength}
          onChange={(e) => setStrength(parseFloat(e.target.value))}
          fullWidth
          sx={{ my: 1 }}
        />
        <TextField
          label="ガイダンス (guidance)"
          type="number"
          value={guidance}
          onChange={(e) => setGuidance(parseFloat(e.target.value))}
          fullWidth
          sx={{ my: 1 }}
        />

        <Button variant="contained" type="submit" disabled={loading} sx={{ mt: 2 }}>
          生成開始
        </Button>
      </form>

      {loading && <LinearProgress sx={{ mt: 2 }} />}

      {/* === 出力表示 === */}
      <Box sx={{ mt: 4 }}>
        {imageUrls.length > 0 && <Typography variant="h6">生成結果:</Typography>}
        <Box
          sx={{
            display: "flex",
            flexWrap: "wrap",
            gap: 2,
            mt: 2,
            justifyContent: "center",
          }}
        >
          {imageUrls.map((url, idx) => (
            <img
              key={idx}
              src={resolveUrl(url)}
              alt={`frame_${idx}`}
              onClick={() => handleImageClick(idx)}
              style={{
                width: "200px",
                borderRadius: "8px",
                boxShadow: "0 0 5px #888",
                cursor: "pointer",
              }}
            />
          ))}
        </Box>
      </Box>

      {/* === プレビュー === */}
      <Dialog open={openDialog} onClose={() => setOpenDialog(false)} maxWidth="lg" fullWidth>
        <DialogTitle sx={{ position: "relative", pr: 5 }}>
          プレビュー
          <IconButton
            onClick={() => setOpenDialog(false)}
            sx={{ position: "absolute", right: 8, top: 8 }}
          >
            <CloseIcon />
          </IconButton>
        </DialogTitle>

        <DialogContent
          sx={{
            position: "relative",
            textAlign: "center",
            backgroundColor: "#000",
          }}
        >
          {imageUrls.length > 1 && (
            <>
              <IconButton
                onClick={handlePrev}
                sx={{
                  position: "absolute",
                  left: 8,
                  top: "50%",
                  transform: "translateY(-50%)",
                  backgroundColor: "rgba(255,255,255,0.6)",
                  zIndex: 10,
                }}
              >
                <ArrowBackIosNewIcon />
              </IconButton>
              <IconButton
                onClick={handleNext}
                sx={{
                  position: "absolute",
                  right: 8,
                  top: "50%",
                  transform: "translateY(-50%)",
                  backgroundColor: "rgba(255,255,255,0.6)",
                  zIndex: 10,
                }}
              >
                <ArrowForwardIosIcon />
              </IconButton>
            </>
          )}

          {imageUrls[currentIndex] && (
            <img
              src={resolveUrl(imageUrls[currentIndex])}
              alt={`frame_${currentIndex}`}
              style={{
                maxWidth: "100%",
                maxHeight: "80vh",
                borderRadius: "8px",
                display: "block",
                margin: "0 auto",
              }}
            />
          )}
          <Typography sx={{ mt: 1, color: "#fff" }}>
            {currentIndex + 1} / {imageUrls.length}（←→キーまたはボタンで移動）
          </Typography>
        </DialogContent>
      </Dialog>
    </Box>
  );
}

