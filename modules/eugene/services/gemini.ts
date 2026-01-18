
import { GoogleGenAI, GenerateContentResponse, Type } from "@google/genai";
import { Message, ModelMode } from "../types";

const API_KEY = process.env.API_KEY || "";

export class GeminiService {
  private ai: GoogleGenAI;

  constructor() {
    this.ai = new GoogleGenAI({ apiKey: API_KEY });
  }

  async performSearch(prompt: string, mode: ModelMode = ModelMode.AUTO): Promise<Partial<Message>> {
    try {
      const isExpert = mode === ModelMode.EXPERT;
      const isThinking = mode === ModelMode.THINKING;

      const response: GenerateContentResponse = await this.ai.models.generateContent({
        model: isExpert ? 'gemini-3-pro-preview' : 'gemini-3-flash-preview',
        contents: prompt,
        config: {
          tools: [{ googleSearch: {} }],
          thinkingConfig: isThinking ? { thinkingBudget: 16000 } : undefined,
        },
      });

      const text = response.text || "I couldn't find an answer to that.";
      const groundingChunks = response.candidates?.[0]?.groundingMetadata?.groundingChunks || [];
      
      const sources = groundingChunks
        .filter(chunk => chunk.web)
        .map(chunk => ({
          title: chunk.web?.title || "Source",
          uri: chunk.web?.uri || ""
        }));

      return {
        content: text,
        sources: sources.length > 0 ? sources : undefined,
        timestamp: new Date()
      };
    } catch (error) {
      console.error("Gemini Search Error:", error);
      return {
        content: "Sorry, I encountered an error while searching. Please check your API key or try again later.",
        timestamp: new Date()
      };
    }
  }

  async generateResponse(history: Message[], mode: ModelMode = ModelMode.AUTO): Promise<Partial<Message>> {
    try {
      const isExpert = mode === ModelMode.EXPERT;
      const isThinking = mode === ModelMode.THINKING;
      
      const contents = history.map(m => ({
        role: m.role === 'user' ? 'user' : 'model',
        parts: [{ text: m.content }]
      }));

      // In this environment we use generateContent directly
      const response: GenerateContentResponse = await this.ai.models.generateContent({
        model: isExpert ? 'gemini-3-pro-preview' : 'gemini-3-flash-preview',
        contents: contents as any,
        config: {
          thinkingConfig: isThinking ? { thinkingBudget: 16000 } : undefined,
          systemInstruction: "You are RankedGPT, a high-performance AI assistant. Provide concise, accurate, and expert-level information."
        }
      });

      return {
        content: response.text || "I'm not sure how to respond to that.",
        timestamp: new Date()
      };
    } catch (error) {
      console.error("Gemini Chat Error:", error);
      return {
        content: "Error generating response.",
        timestamp: new Date()
      };
    }
  }
}

export const geminiService = new GeminiService();
