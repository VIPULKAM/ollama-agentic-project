# How to Get API Keys for Tool Testing

Your LangGraph agent is **fully implemented and working**! You just need a valid API key with credits/quota to test tool execution. Here are your options:

---

## 🆓 Option 1: Gemini (Google) - FREE Tier Available! (RECOMMENDED)

### Why Gemini?
- ✅ **Free tier** with generous quota
- ✅ Supports tool calling (function calling)
- ✅ Fast and capable
- ✅ No credit card required initially

### Steps:

1. **Visit Google AI Studio**
   ```
   https://aistudio.google.com/app/apikey
   ```

2. **Sign in with Google Account**
   - Use your existing Google account
   - Or create a new one (free)

3. **Create API Key**
   - Click "Create API Key"
   - Choose "Create API key in new project" or use existing project
   - Copy the generated key

4. **Update .env File**
   ```bash
   # Open .env in your editor
   nano .env

   # Replace the expired key with new one
   GOOGLE_API_KEY="your-new-key-here"

   # Save and exit
   ```

5. **Test Your Agent**
   ```bash
   python test_claude_tools.py
   ```

### Gemini Free Tier Limits:
- 15 requests per minute
- 1 million tokens per minute
- 1,500 requests per day
- **More than enough for testing!**

---

## 💳 Option 2: Claude (Anthropic) - $5 Minimum

### Why Claude?
- ✅ Excellent at tool use
- ✅ High quality responses
- ✅ Great for production use
- ❌ Requires payment ($5 minimum)

### Steps:

1. **Visit Anthropic Console**
   ```
   https://console.anthropic.com/settings/billing
   ```

2. **Add Credits**
   - Click "Add Credits"
   - Minimum: $5 USD
   - Accept major credit cards

3. **Your Key is Already Set**
   - Already in .env: `ANTHROPIC_API_KEY="sk-ant-api03..."`
   - No need to update it

4. **Test Your Agent**
   ```bash
   # Change provider in .env
   LLM_PROVIDER=claude

   # Run tests
   python test_claude_tools.py
   ```

### Claude Pricing:
- Haiku (what you're using): $0.25 per million input tokens
- Very affordable for testing

---

## 🔧 Option 3: Use Ollama (No API Keys) - Limited Functionality

### Why Ollama?
- ✅ Completely free
- ✅ Works offline
- ✅ Already running on your machine
- ❌ qwen2.5-coder:1.5b doesn't support tools

### Steps:

1. **Set Tools to False**
   ```bash
   # Edit .env
   ENABLE_TOOLS=False
   ```

2. **Run the Agent**
   ```bash
   python main.py
   ```

3. **Conversational Mode**
   - Works great for Q&A
   - No file operations or RAG
   - Good for general coding questions

---

## 🎯 Recommended: Get Gemini Key (5 minutes, FREE)

**Fastest path to test your LangGraph agent:**

```bash
# 1. Get key from: https://aistudio.google.com/app/apikey
# 2. Update .env: GOOGLE_API_KEY="your-key"
# 3. Run: python test_claude_tools.py
```

That's it! Your agent will execute tools, read files, write files, and search code. 🚀

---

## 📋 Quick Test After Getting Key

```bash
# Test simple query
python -c "
from src.agent.agent import CodingAgent
agent = CodingAgent(provider='gemini', temperature=0.1)
print(agent.ask('What is 2+2?'))
"

# If you see '4' - SUCCESS! ✅
# Then run full test suite:
python test_claude_tools.py
```

---

## ❓ Need Help?

If you get an error after adding a key:

1. **Check .env format**: No extra quotes or spaces
2. **Reload environment**: Restart terminal or `source .env`
3. **Verify key**: Test at API provider's website first
4. **Check limits**: Make sure you're within free tier limits

---

## 🎉 What You'll See Working

Once you have a valid key:

✅ Agent answers questions (with and without tools)
✅ Agent reads files on demand
✅ Agent writes files with confirmation
✅ Agent lists directories
✅ Agent searches code with regex
✅ Agent remembers conversation context
✅ Multi-turn dialogues with tool use

**Your LangGraph implementation is ready!** Just add a key and enjoy. 🚀
