# realtime-voice-order-assisstant
An End-to-End Voice AI ordering system for restaurants, featuring real-time interaction and RAG menu knowledge base.
# 🍽️ realtime-voice-order-assisstant (餐飲智慧語音點餐服務員)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Architecture](https://img.shields.io/badge/architecture-Cloud%20Native-orange)

## 📌 Project Overview
This project is an AI-driven Voice SaaS solution designed to tackle labor shortages in the food and beverage industry. By leveraging an **End-to-End Multimodal Large Language Model (e.g., `gpt-realtime` integration)** and a seamless Web Audio interface, customers can naturally interact, order, and inquire without downloading any apps.

## 🚀 Core Features
* **Real-Time Voice Interaction:** Zero-latency conversation using Web Audio API and WebSocket streaming.
* **Edge AI VAD (Voice Activity Detection):** Lightweight 2-layer CNN + LSTM for efficient audio feature extraction directly on the client side.
* **Menu Knowledge Retrieval (RAG):** Integrates **Supabase** as a vector store to ensure highly accurate menu, pricing, and promotional information.
* **Automated Agent Workflows:** Utilizes **n8n** to orchestrate intention routing, order confirmation, and function-calling seamlessly.

## 🛠️ Tech Stack
* **Frontend:** HTML5 Web Audio API, JavaScript
* **Backend:** Python `asyncio`, WebSocket
* **AI / Models:** End-to-End Speech Model (`gpt-realtime`), Neural Vocoder
* **Database:** Supabase (Vector Store for RAG)
* **Workflow Automation:** n8n 
* **Deployment:** Docker, WSL (Windows Subsystem for Linux), Cloudflare Tunnel (for secure local testing and public routing)

## ⚙️ Quick Start (Development Environment)

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YourUsername/realtime-voice-order-assisstant.git](https://github.com/YourUsername/realtime-voice-order-assisstant.git)
