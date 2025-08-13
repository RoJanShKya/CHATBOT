import streamlit as st
import os
import pickle
from datetime import datetime
import tempfile
from typing import List, Dict, Any

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI

# Set page config
st.set_page_config(
    page_title="AI PDF ChatBot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Essential CSS for clean UI
st.markdown("""
<style>
/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none;}

/* Main container */
.main .block-container {
    padding-top: 1rem;
    padding-bottom: 120px;
    max-width: 100%;
}

/* Chat container - fixed height with auto scroll */
.chat-container {
    height: 500px;
    overflow-y: auto;
    padding: 20px;
    background: #f8f9fa;
    border-radius: 10px;
    margin-bottom: 20px;
    border: 1px solid #e0e0e0;
    scroll-behavior: smooth;
}

/* Auto scroll to bottom */
.chat-container::-webkit-scrollbar {
    width: 6px;
}

.chat-container::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 3px;
}

.chat-container::-webkit-scrollbar-thumb {
    background: #c1c1c1;
    border-radius: 3px;
}

/* User message */
.user-message {
    display: flex;
    justify-content: flex-end;
    margin: 10px 0;
}

.user-message .message-content {
    background: #007bff;
    color: white;
    padding: 12px 16px;
    border-radius: 15px 15px 5px 15px;
    max-width: 70%;
    word-wrap: break-word;
    font-size: 14px;
    line-height: 1.4;
}

/* Bot message */
.bot-message {
    display: flex;
    justify-content: flex-start;
    margin: 10px 0;
}

.bot-message .message-content {
    background: #e9ecef;
    color: #333;
    padding: 12px 16px;
    border-radius: 15px 15px 15px 5px;
    max-width: 70%;
    word-wrap: break-word;
    font-size: 14px;
    line-height: 1.4;
}

/* Input container - fixed at bottom */
.input-container {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: white;
    padding: 15px 20px;
    border-top: 1px solid #e0e0e0;
    z-index: 1000;
    box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
}

/* Sidebar styling */
.sidebar .sidebar-content {
    background: #f8f9fa;
}

/* File info styling */
.file-info {
    background: #e3f2fd;
    padding: 10px;
    border-radius: 8px;
    margin: 5px 0;
    border-left: 4px solid #2196f3;
}

/* Status indicators */
.status-success {
    color: #28a745;
    font-weight: bold;
}

.status-error {
    color: #dc3545;
    font-weight: bold;
}

.status-warning {
    color: #ffc107;
    font-weight: bold;
}

/* Greeting message */
.greeting {
    text-align: center;
    padding: 40px 20px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 15px;
    margin: 20px 0;
    font-size: 1.5em;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

class PDFChatBot:
    def __init__(self):
        self.setup_directories()
        self.initialize_session_state()
        self.setup_models()
        
    def setup_directories(self):
        """Create necessary directories for file storage"""
        self.base_dir = "pdf_chatbot_data"
        self.vectorstore_dir = os.path.join(self.base_dir, "vectorstore")
        self.pdfs_dir = os.path.join(self.base_dir, "pdfs")
        self.processed_files_db = os.path.join(self.base_dir, "processed_files.pkl")
        
        # Create directories if they don't exist
        for dir_path in [self.base_dir, self.vectorstore_dir, self.pdfs_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def initialize_session_state(self):
        """Initialize all session state variables"""
        defaults = {
            'chat_history': [],
            'processed_files': self.load_processed_files(),
            'vectorstore': None,
            'chain': None,
            'models_ready': False,
            'greeting_shown': False,
            'last_query': "",
            'processing_query': False
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def setup_models(self):
        """Setup embeddings and LLM models"""
        try:
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Initialize Gemini LLM
            GEMINI_API_KEY = "AIzaSyBtnLy5u7SIsdaQKYPFywswPwLCQzSvyFw"
            
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-latest",
                google_api_key=GEMINI_API_KEY,
                temperature=0.3,
                max_tokens=1500
            )
            
            # Test API connection
            test_response = self.llm.predict("Hello")
            st.session_state.models_ready = True
            
        except Exception as e:
            st.error(f"❌ Error setting up models: {str(e)}")
            st.session_state.models_ready = False
    
    def load_processed_files(self) -> Dict[str, Any]:
        """Load processed files database"""
        if os.path.exists(self.processed_files_db):
            try:
                with open(self.processed_files_db, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                return {}
        return {}
    
    def save_processed_files(self):
        """Save processed files database"""
        try:
            with open(self.processed_files_db, 'wb') as f:
                pickle.dump(st.session_state.processed_files, f)
        except Exception as e:
            st.error(f"Error saving processed files: {str(e)}")
    
    def is_file_processed(self, filename: str, file_size: int) -> bool:
        """Check if file is already processed"""
        if filename in st.session_state.processed_files:
            stored_info = st.session_state.processed_files[filename]
            return stored_info.get('size') == file_size
        return False
    
    def process_pdfs(self, uploaded_files) -> tuple[bool, str]:
        """Process uploaded PDF files"""
        if not uploaded_files:
            return False, "No files uploaded"
        
        try:
            documents = []
            new_files_count = 0
            skipped_files = []
            
            for uploaded_file in uploaded_files:
                file_size = len(uploaded_file.getbuffer())
                
                # Skip if already processed
                if self.is_file_processed(uploaded_file.name, file_size):
                    skipped_files.append(uploaded_file.name)
                    continue
                
                # Save file locally
                file_path = os.path.join(self.pdfs_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load and process PDF
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                documents.extend(docs)
                
                # Store file info
                st.session_state.processed_files[uploaded_file.name] = {
                    'processed_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'size': file_size,
                    'pages': len(docs),
                    'path': file_path,
                    'status': 'processed'
                }
                new_files_count += 1
            
            # Process new documents
            if documents:
                # Split documents into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                splits = text_splitter.split_documents(documents)
                
                # Create or update vectorstore
                vectorstore_path = os.path.join(self.vectorstore_dir, "faiss_index")
                
                if st.session_state.vectorstore is None:
                    # Create new vectorstore
                    vectorstore = FAISS.from_documents(splits, self.embeddings)
                else:
                    # Add to existing vectorstore
                    vectorstore = st.session_state.vectorstore
                    vectorstore.add_documents(splits)
                
                # Save vectorstore
                vectorstore.save_local(vectorstore_path)
                st.session_state.vectorstore = vectorstore
                
                # Save processed files info
                self.save_processed_files()
                
                # Create/update conversation chain
                self.create_conversation_chain()
            
            # Prepare status message
            messages = []
            if new_files_count > 0:
                messages.append(f"✅ Processed {new_files_count} new files")
            if skipped_files:
                messages.append(f"⏭️ Skipped {len(skipped_files)} already processed files")
            
            return True, " | ".join(messages) if messages else "All files already processed"
            
        except Exception as e:
            return False, f"Error processing PDFs: {str(e)}"
    
    def load_existing_vectorstore(self):
        """Load existing vectorstore if available"""
        vectorstore_path = os.path.join(self.vectorstore_dir, "faiss_index")
        if os.path.exists(vectorstore_path):
            try:
                st.session_state.vectorstore = FAISS.load_local(
                    vectorstore_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                self.create_conversation_chain()
                return True
            except Exception as e:
                st.warning(f"Could not load existing vectorstore: {str(e)}")
                return False
        return False
    
    def create_conversation_chain(self):
        """Create conversation chain"""
        if st.session_state.vectorstore and hasattr(self, 'llm'):
            try:
                memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True
                )
                
                st.session_state.chain = ConversationalRetrievalChain.from_llm(
                    llm=self.llm,
                    retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3}),
                    memory=memory
                )
            except Exception as e:
                st.error(f"Error creating conversation chain: {str(e)}")
    
    def render_sidebar(self):
        """Render sidebar with file management"""
        with st.sidebar:
            st.title("🤖 AI PDF ChatBot")
            
            # Model status
            if st.session_state.models_ready:
                st.success("✅ Models Ready")
            else:
                st.error("❌ Models Not Ready")
            
            st.markdown("---")
            
            # File upload section
            st.subheader("📁 Upload PDFs")
            uploaded_files = st.file_uploader(
                "Choose PDF files",
                type="pdf",
                accept_multiple_files=True,
                key="pdf_uploader"
            )
            
            if st.button("🚀 Process PDFs", type="primary", use_container_width=True):
                if uploaded_files:
                    with st.spinner("Processing PDFs..."):
                        success, message = self.process_pdfs(uploaded_files)
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
                        st.rerun()
                else:
                    st.warning("Please upload PDF files first")
            
            st.markdown("---")
            
            # Processed files display
            st.subheader("📋 Processed Files")
            if st.session_state.processed_files:
                for filename, info in st.session_state.processed_files.items():
                    st.markdown(f"""
                    <div class="file-info">
                        <strong>📄 {filename}</strong><br>
                        <small>📅 {info['processed_at']}</small><br>
                        <small>📖 {info['pages']} pages</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No files processed yet")
            
            st.markdown("---")
            
            # Clear conversation button
            if st.session_state.chat_history:
                if st.button("🗑️ Clear Chat", use_container_width=True):
                    st.session_state.chat_history = []
                    st.session_state.greeting_shown = False
                    st.session_state.last_query = ""
                    st.session_state.processing_query = False
                    st.rerun()
    
    def render_chat_interface(self):
        """Render main chat interface"""
        # Load existing vectorstore on startup
        if st.session_state.vectorstore is None:
            self.load_existing_vectorstore()
        
        # Greeting message 
        if not st.session_state.greeting_shown and not st.session_state.chat_history:
            st.markdown("""
            <div class="greeting">
                👋 Welcome to AI PDF ChatBot!<br>
                <small>Upload PDFs and start chatting!</small>
            </div>
            """, unsafe_allow_html=True)
            st.session_state.greeting_shown = True
        
        # Chat history container
        chat_container = st.container()
        
        with chat_container:
            if st.session_state.chat_history:
                st.markdown('<div class="chat-container" id="chat-container">', unsafe_allow_html=True)
                
                for message in st.session_state.chat_history:
                    if message['type'] == 'user':
                        st.markdown(f"""
                        <div class="user-message">
                            <div class="message-content">
                                {message['content']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="bot-message">
                            <div class="message-content">
                                {message['content']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Auto-scroll to bottom
                st.markdown("""
                <script>
                var chatContainer = document.getElementById('chat-container');
                if (chatContainer) {
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }
                </script>
                """, unsafe_allow_html=True)
        
        # Input area (fixed at bottom)
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([5, 1])
        
        with col1:
            query = st.text_input(
                "",
                key="query_input",
                placeholder="Ask me anything about your PDFs...",
                label_visibility="collapsed"
            )
        
        with col2:
            send_button = st.button("Send", type="primary", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Process query (prevent duplicates)
        if (send_button or query) and query.strip():
            if (query.strip() != st.session_state.last_query and 
                not st.session_state.processing_query):
                
                if st.session_state.chain is not None:
                    # Set processing flags
                    st.session_state.processing_query = True
                    st.session_state.last_query = query.strip()
                    
                    # Add user message
                    st.session_state.chat_history.append({
                        'type': 'user',
                        'content': query.strip(),
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # Get bot response
                    try:
                        with st.spinner(" Thinking..."):
                            response = st.session_state.chain({"question": query.strip()})
                            answer = response['answer']
                            
                            # Add bot response
                            st.session_state.chat_history.append({
                                'type': 'bot',
                                'content': answer,
                                'timestamp': datetime.now().isoformat()
                            })
                            
                    except Exception as e:
                        st.session_state.chat_history.append({
                            'type': 'bot',
                            'content': f"❌ Sorry, I encountered an error: {str(e)}",
                            'timestamp': datetime.now().isoformat()
                        })
                    
                    # Reset processing state
                    st.session_state.processing_query = False
                    st.rerun()
                    
                else:
                    st.warning("⚠️ Please upload and process PDF files first!")
    
    def run(self):
        """Run the main application"""
        self.render_sidebar()
        self.render_chat_interface()

# Initialize and run the app
if __name__ == "__main__":
    try:
        chatbot = PDFChatBot()
        chatbot.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please refresh the page and try again.")
        st.info("IF THE ERROR STILL CONTINUES PLEASE REPORT IT TO THE DEVELOPER.")