import React, { useState } from "react";
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
  IconButton,
} from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";

export default function GenerateForm() {
  const [image1, setImage1] = useState(null);
  const [image2, setImage2] = useState(null);
  const [frames, setFrames] = useState(8);
  const [t0, setT0] = useState(5.0);
  const [lineart, setLineart] = useState(false);
  const [denoise, setDenoise] = useState(false);
  const [hybrid, setHybrid] = useState(false);
  const [imageUrls, setImageUrls] = useState([]);
  const [loading, setLoading] = useState(false);
  const [openDialog, setOpenDialog] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null);

  const API_BASE = "http://43.207.92.186:8000";

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
    formData.append("lineart", hybrid ? true : lineart);
    formData.append("denoise", hybrid ? true : denoise);

    try {
      setLoading(true);
      const res = await axios.post(`${API_BASE}/generate`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      if (res.data.image_urls) setImageUrls(res.data.image_urls);
    } catch (err) {
      console.error(err);
      alert("生成中にエラーが発生しました。");
    } finally {
      setLoading(false);
    }
  };

  const handleImageClick = (url) => {
    setSelectedImage(url);
    setOpenDialog(true);
  };

  return (
    <Box sx={{ maxWidth: 600, mx: "auto", mt: 4, textAlign: "center" }}>
      <Typography variant="h4" gutterBottom>
        Time Reversal Generator
      </Typography>

      <form onSubmit={handleSubmit}>
        <Box sx={{ my: 2 }}>
          <Typography>画像A</Typography>
          <input type="file" accept="image/*" onChange={(e) => setImage1(e.target.files[0])} />
        </Box>
        <Box sx={{ my: 2 }}>
          <Typography>画像B</Typography>
          <input type="file" accept="image/*" onChange={(e) => setImage2(e.target.files[0])} />
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
          control={<Checkbox checked={lineart} onChange={(e) => setLineart(e.target.checked)} />}
          label="線画抽出モード"
        />
        <FormControlLabel
          control={<Checkbox checked={denoise} onChange={(e) => setDenoise(e.target.checked)} />}
          label="ノイズ除去モード"
        />
        <FormControlLabel
          control={<Checkbox checked={hybrid} onChange={(e) => setHybrid(e.target.checked)} />}
          label="線画＋ノイズ除去ハイブリッド"
        />

        <Button variant="contained" type="submit" disabled={loading} sx={{ mt: 2 }}>
          生成開始
        </Button>
      </form>

      {loading && <LinearProgress sx={{ mt: 2 }} />}

      <Box sx={{ mt: 4 }}>
        {imageUrls.length > 0 && <Typography variant="h6">生成結果:</Typography>}
        <Box sx={{ display: "flex", flexWrap: "wrap", gap: 2, mt: 2, justifyContent: "center" }}>
          {imageUrls.map((url, idx) => (
            <img
              key={idx}
              src={`${API_BASE}${url}`}
              alt={`frame_${idx}`}
              onClick={() => handleImageClick(`${API_BASE}${url}`)}
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

      <Dialog open={openDialog} onClose={() => setOpenDialog(false)} maxWidth="lg">
        <Box sx={{ position: "absolute", top: 8, right: 8 }}>
          <IconButton onClick={() => setOpenDialog(false)}>
            <CloseIcon />
          </IconButton>
        </Box>
        <DialogContent>
          {selectedImage && (
            <img
              src={selectedImage}
              alt="拡大表示"
              style={{
                maxWidth: "100%",
                maxHeight: "80vh",
                borderRadius: "8px",
                display: "block",
                margin: "0 auto",
              }}
            />
          )}
        </DialogContent>
      </Dialog>
    </Box>
  );
}

