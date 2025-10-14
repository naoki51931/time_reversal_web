import React, { useState } from "react";
import axios from "axios";
import {
  Box,
  Button,
  TextField,
  Typography,
  CircularProgress,
  Stack,
} from "@mui/material";

const API_URL = process.env.REACT_APP_API_URL || "http://43.207.92.186:8000";

export default function GenerateForm() {
  const [image1, setImage1] = useState(null);
  const [image2, setImage2] = useState(null);
  const [frames, setFrames] = useState(2);
  const [t0, setT0] = useState(5.0);
  const [loading, setLoading] = useState(false);
  const [resultUrls, setResultUrls] = useState([]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!image1 || !image2) {
      alert("2枚の画像を選択してください。");
      return;
    }

    setLoading(true);
    setResultUrls([]);

    try {
      const formData = new FormData();
      formData.append("image_1", image1);
      formData.append("image_2", image2);
      formData.append("frames", frames);
      formData.append("t0", t0);

      console.log("[Client] Sending request to:", `${API_URL}/generate`);
      const res = await axios.post(`${API_URL}/generate`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      console.log("[Client] Response:", res.data);

      if (res.data.image_urls) {
        setResultUrls(res.data.image_urls);
      } else if (res.data.image_url) {
        setResultUrls([res.data.image_url]);
      } else {
        alert("画像URLが返されませんでした。");
      }
    } catch (err) {
      console.error("Error during generation:", err);
      alert("生成中にエラーが発生しました。");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box
      sx={{
        maxWidth: 600,
        mx: "auto",
        mt: 6,
        p: 4,
        border: "1px solid #ccc",
        borderRadius: 2,
        backgroundColor: "#fafafa",
      }}
    >
      <Typography variant="h5" gutterBottom align="center">
        Time Reversal Generator
      </Typography>

      <form onSubmit={handleSubmit}>
        <Stack spacing={2}>
          <Box>
            <Typography variant="subtitle1">画像A（始まり）</Typography>
            <input
              type="file"
              accept="image/*"
              onChange={(e) => setImage1(e.target.files[0])}
            />
          </Box>

          <Box>
            <Typography variant="subtitle1">画像B（終わり）</Typography>
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
          />

          <TextField
            label="t0 (時間パラメータ)"
            type="number"
            value={t0}
            onChange={(e) => setT0(e.target.value)}
            fullWidth
          />

          <Button
            type="submit"
            variant="contained"
            color="primary"
            disabled={loading}
          >
            {loading ? "生成中..." : "生成開始"}
          </Button>
        </Stack>
      </form>

      {loading && (
        <Box textAlign="center" mt={3}>
          <CircularProgress />
        </Box>
      )}

      {!loading && resultUrls.length > 0 && (
        <Box mt={4}>
          <Typography variant="h6">生成結果</Typography>
          <Stack spacing={2} mt={2}>
            {resultUrls.map((url, idx) => (
              <Box key={idx} textAlign="center">
                <Typography variant="subtitle2">Frame {idx}</Typography>
                <img
                  src={url}
                  alt={`Generated frame ${idx}`}
                  style={{
                    maxWidth: "100%",
                    borderRadius: 8,
                    border: "1px solid #ccc",
                  }}
                />
              </Box>
            ))}
          </Stack>
        </Box>
      )}
    </Box>
  );
}

