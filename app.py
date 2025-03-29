import validators, streamlit as st
import os
import re
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.schema import Document

#streamlit app
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜LangChain: Summarize Text From YouTube or Website")
st.subheader('Summarize URL')

#Set the GROQ API key
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("groq_api_key")

generic_url=st.text_input("URL", label_visibility="hidden")

#Intialize the model
llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)

prompt_template="""
Provide a summary of the following content in 300 words:
Content:{text}
"""
prompt=PromptTemplate(template=prompt_template, input_variables=['text'])

def extract_video_id(url):
    """Extracts the video ID from various YouTube URL formats"""
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",  # Standard YouTube URL
        r"(?:youtu\.be\/|youtube\.com\/embed\/|youtube\.com\/v\/)([0-9A-Za-z_-]{11})"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_youtube_transcript(video_id):
    """Fetches transcript from YouTube"""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([entry["text"] for entry in transcript])
        return text
    except Exception as e:
        return f"❌ Error: {e}"
    
MAX_TOKENS = 10000
def truncate_text(text, max_tokens=MAX_TOKENS):
    words = text.split()
    return " ".join(words[:max_tokens])

if st.button("Summarize the Content from YT or Website"):
    #Validate all the inputs
    if not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid Url.It can may be a YT video utl or website url")   
    else:
        try:
            with st.spinner("Waiting...."):
                #loading the website or yt video data
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    video_id = extract_video_id(generic_url)
                    loader=get_youtube_transcript(video_id)
                    docs=[Document(page_content=loader)]
                else:
                    loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                                                 headers={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"})
                    docs=loader.load()

                if docs:
                    docs[0].page_content = truncate_text(docs[0].page_content)

                #Chain for Summarization
                chain=load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                summary_text = chain.invoke(docs)["output_text"]
                st.success("Summary Generated Successfully!")
                st.write(summary_text)
        
        except Exception as e:
            st.exception(f"Exception:{e}")