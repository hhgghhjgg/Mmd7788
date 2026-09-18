import express from "express";
import path from "path";
import { fileURLToPath } from "url";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function startServer() {
  const app = express();
  const PORT = 3000;

  // JSON and URL-encoded body parser with 50MB limit for high-res images and manga chunks
  app.use(express.json({ limit: "50mb" }));
  app.use(express.urlencoded({ extended: true, limit: "50mb" }));

  let aiClient: GoogleGenAI | null = null;
  function getGenAI(): GoogleGenAI {
    if (!aiClient) {
      const apiKey = process.env.GEMINI_API_KEY;
      if (!apiKey) {
        throw new Error("GEMINI_API_KEY is not configured in the environment.");
      }
      aiClient = new GoogleGenAI({
        apiKey,
        httpOptions: {
          headers: {
            "User-Agent": "aistudio-build",
          },
        },
      });
    }
    return aiClient;
  }

  // Health check endpoint
  app.get("/api/health", (_req, res) => {
    res.json({ status: "ok" });
  });

  // Server-side Gemini proxy endpoint
  app.post("/api/gemini/generate", async (req, res) => {
    try {
      const { parts, config, model } = req.body;
      if (!parts || !Array.isArray(parts) || parts.length === 0) {
        return res.status(400).json({ error: "Missing or invalid 'parts' array" });
      }

      const ai = getGenAI();

      const formattedParts = parts.map((part: any) => {
        if (part.inlineData) {
          return {
            inlineData: {
              mimeType: part.inlineData.mimeType,
              data: part.inlineData.data,
            },
          };
        }
        return {
          text: String(part.text || ""),
        };
      });

      // Requested model: default to Gemini 3 Flash (gemini-3-flash-preview)
      const requestedModel = typeof model === "string" && model.trim() 
        ? model.trim() 
        : "gemini-3-flash-preview";

      let text = "";
      try {
        const response = await ai.models.generateContent({
          model: requestedModel,
          contents: { parts: formattedParts },
          config: {
            temperature: typeof config?.temperature === "number" ? config.temperature : 0.3,
            topP: typeof config?.topP === "number" ? config.topP : 0.95,
            topK: typeof config?.topK === "number" ? config.topK : 40,
          },
        });
        text = response.text || "";
      } catch (modelError: any) {
        // If the requested model (e.g., gemini-3.8-flash) times out or errors, try fast reliable fallback
        if (requestedModel !== "gemini-2.5-flash-lite") {
          console.warn(`Model ${requestedModel} failed, falling back to gemini-2.5-flash-lite:`, modelError.message);
          const fallbackResponse = await ai.models.generateContent({
            model: "gemini-2.5-flash-lite",
            contents: { parts: formattedParts },
            config: {
              temperature: typeof config?.temperature === "number" ? config.temperature : 0.3,
              topP: typeof config?.topP === "number" ? config.topP : 0.95,
              topK: typeof config?.topK === "number" ? config.topK : 40,
            },
          });
          text = fallbackResponse.text || "";
        } else {
          throw modelError;
        }
      }

      return res.json({ text });
    } catch (error: any) {
      console.error("Gemini Generation Error:", error);
      const errorMessage = error?.message || "Error generating content with Gemini.";
      return res.status(500).json({ error: errorMessage });
    }
  });

  // Vite integration: development middleware or static production serving
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (_req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
