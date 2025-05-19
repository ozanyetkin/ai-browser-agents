#!/usr/bin/env python
# coding: utf-8

# # L2: Building a Simple Web Agent

# <p style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px"> ⏳ <b>Note <code>(Kernel Starting)</code>:</b> This notebook takes about 30 seconds to be ready to use. You may start and watch the video while you wait.</p>

# In[ ]:


import asyncio
import json
import os
import nest_asyncio
import pprint
import base64
from io import BytesIO
import pandas as pd
from playwright.async_api import async_playwright
from openai import OpenAI
from PIL import Image
from tabulate import tabulate
from IPython.display import display, HTML, Markdown
from pydantic import BaseModel
from dotenv import load_dotenv


# <div style="background-color:#fff6ff; padding:13px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px">
# <p> 💻 &nbsp; <b>Access <code>requirements.txt</code> and <code>helper.py</code> files:</b> 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Open"</em>.
# 
# <p> ⬇ &nbsp; <b>Download Notebooks:</b> 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Download as"</em> and select <em>"Notebook (.ipynb)"</em>.</p>
# 
# <p> 📒 &nbsp; For more help, please see the <em>"Appendix – Tips, Help, and Download"</em> Lesson.</p>
# </div>

# In[ ]:


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
nest_asyncio.apply()


# ## WebScraper Agent

# In[ ]:


class WebScraperAgent:
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.page = None

    async def init_browser(self):
      self.playwright = await async_playwright().start()
      self.browser = await self.playwright.chromium.launch(
          headless=True,
          args=[
              "--disable-dev-shm-usage",
              "--no-sandbox",
              "--disable-setuid-sandbox",
              "--disable-accelerated-2d-canvas",
              "--disable-gpu",
              "--no-zygote",
              "--disable-audio-output",
              "--disable-software-rasterizer",
              "--disable-webgl",
              "--disable-web-security",
              "--disable-features=LazyFrameLoading",
              "--disable-features=IsolateOrigins",
              "--disable-background-networking"
          ]
      )
      self.page = await self.browser.new_page()

    async def scrape_content(self, url):
        if not self.page or self.page.is_closed():
            await self.init_browser()
        await self.page.goto(url, wait_until="load")
        await self.page.wait_for_timeout(2000)  # Wait for dynamic content
        return await self.page.content()

    async def take_screenshot(self, path="screenshot.png"):
        await self.page.screenshot(path=path, full_page=True)
        return path
    async def screenshot_buffer(self):
        screenshot_bytes = await self.page.screenshot(type="png", full_page=False)
        return screenshot_bytes

    async def close(self):
        await self.browser.close()
        await self.playwright.stop()
        self.playwright = None
        self.browser = None
        self.page = None


# In[ ]:


scraper = WebScraperAgent()


# ## Structured Data Format

# In[ ]:


class DeeplearningCourse(BaseModel):
    title: str
    description: str
    presenter: list[str]
    imageUrl: str
    courseURL: str

class DeeplearningCourseList(BaseModel):
    courses: list[DeeplearningCourse]


# ## LLM Client for Open AI

# In[ ]:


async def process_with_llm(html, instructions, truncate = False):
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini-2024-07-18",
        messages=[{
            "role": "system",
            "content": f"""
            You are an expert web scraping agent. Your task is to:
            Extract relevant information from this HTML to JSON 
            following these instructions:
            {instructions}
            
            Extract the title, description, presenter, 
            the image URL and course URL for each of 
            all the courses for the deeplearning.ai website

            Return ONLY valid JSON, no markdown or extra text."""
        }, {
            "role": "user",
            "content": html[:150000]  # Truncate to stay under token limits
        }],
        temperature=0.1,
        response_format=DeeplearningCourseList,
        )
    return completion.choices[0].message.parsed


# In[ ]:


async def webscraper(target_url, instructions):
    result = None
    try:
        # Scrape content and capture screenshot
        print("Extracting HTML Content \n")
        html_content = await scraper.scrape_content(target_url)

        print("Taking Screenshot \n")
        screenshot = await scraper.screenshot_buffer()
        # Process content

        print("Processing..")
        result: DeeplearningCourseList = await process_with_llm(html_content, instructions, False)
        print("\nGenerated Structured Response")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    finally:
        await scraper.close()
    return result, screenshot


# ## Example 1

# In[ ]:


target_url = "https://www.deeplearning.ai/courses"  # Deeplearning AI courses
base_url="https://deeplearning.ai"


# In[ ]:


instructions = """
    Get all the courses
"""
result, screenshot = asyncio.run(webscraper(target_url, instructions))


# In[ ]:


asyncio.run(visualizeCourses(result=result, 
                             screenshot=screenshot, 
                             target_url=target_url, 
                             instructions=instructions, 
                             base_url=base_url))


# ## Example with RAG courses

# In[ ]:


subject = "Retrieval Augmented Generation (RAG) "

instructions = f"""
Read the description of the courses and only 
provide the three courses that are about {subject}. 
Make sure that we don't have any other
cources in the output
"""
result, screenshot = asyncio.run(webscraper(target_url, instructions))


# In[ ]:


asyncio.run(visualizeCourses(result=result, 
                             screenshot=screenshot, 
                             target_url=target_url, 
                             instructions=instructions, 
                             base_url=base_url))


# ## Challenges in the web agents

# In[ ]:


subject = "Retrieval Augmented Generation (RAG) "
instructions = f"""
Can you get the summary of the top course on
{subject} provide the learnings from it
"""
result, screenshot = asyncio.run(webscraper(target_url, instructions))


# In[ ]:


asyncio.run(visualizeCourses(result=result,
                             screenshot=screenshot,
                             target_url=target_url,
                             instructions=instructions,
                             base_url=base_url))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




