import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import Message from './Message';
import { ThreeDots } from 'react-loader-spinner';

const languages = [
  { code: 'hi-IN', name: 'Hindi' },
  { code: 'bn-IN', name: 'Bengali' },
  { code: 'kn-IN', name: 'Kannada' },
  { code: 'ml-IN', name: 'Malayalam' },
  { code: 'mr-IN', name: 'Marathi' },
  { code: 'od-IN', name: 'Odia' },
  { code: 'pa-IN', name: 'Punjabi' },
  { code: 'ta-IN', name: 'Tamil' },
  { code: 'te-IN', name: 'Telugu' },
  { code: 'gu-IN', name: 'Gujarati' },
];



function App() {
  const [question, setQuestion] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [selectedLanguage, setSelectedLanguage] = useState('hi-IN'); // Default to Hindi
  const audioRef = useRef(null);
  const messagesEndRef = useRef(null); // Reference for scrolling

  useEffect(() => {
    setMessages([{ text: "Hi, how can I help you?", type: 'ai', id: Date.now() }]);
  }, []);

  useEffect(() => {
    // Scroll to the bottom of the messages whenever messages change
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleInputChange = (e) => {
    setQuestion(e.target.value);
  };

  const handleFormSubmit = async (e) => {
    e.preventDefault();
    if (question.trim() === '') return;

    setMessages([...messages, { text: question, type: 'user', id: Date.now() }]);
    setQuestion('');
    setLoading(true);
    setError('');

    try {
      const response = await axios.post('http://localhost:8000/agent', {
        question,
      });
      setMessages((prevMessages) => [
        ...prevMessages,
        { text: response.data.response, type: response.data.type, id: response.data.id },
      ]);
    } catch (err) {
      setError('An error occurred while fetching the answer.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleLanguageChange = (e) => {
    setSelectedLanguage(e.target.value);
    // Stop any currently playing audio
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
    }
  };

  const handleTextToSpeech = async (id, text) => {
    if (audioRef.current) {
      audioRef.current.pause(); // Stop any currently playing audio
    }

    setMessages((prevMessages) =>
      prevMessages.map((msg) =>
        msg.id === id ? { ...msg, loading: true } : msg
      )
    );

    try {
      const response = await axios.post('http://localhost:8000/text-to-speech/', {
        question: text,
        language: selectedLanguage, // Use the selected language
      });

      const audioElement = new Audio(`data:audio/wav;base64,${response.data.audio}`);
      audioRef.current = audioElement;

      audioElement.play().then(() => {
        // When the audio is played, update the message state
        setMessages((prevMessages) =>
          prevMessages.map((msg) =>
            msg.id === id ? { ...msg, audio: response.data.audio, loading: false } : msg
          )
        );
      }).catch((error) => {
        console.error('Error playing audio:', error);
        setMessages((prevMessages) =>
          prevMessages.map((msg) =>
            msg.id === id ? { ...msg, loading: false } : msg
          )
        );
      });
    } catch (err) {
      setError('An error occurred while converting the text to speech.');
      console.error(err);
      setMessages((prevMessages) =>
        prevMessages.map((msg) =>
          msg.id === id ? { ...msg, loading: false } : msg
        )
      );
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Rag Chatbot</h1>
        <div className="language-dropdown">
          <label htmlFor="language-select">Select Language:</label>
          <select id="language-select" value={selectedLanguage} onChange={handleLanguageChange}>
            {languages.map((lang) => (
              <option key={lang.code} value={lang.code}>
                {lang.name}
              </option>
            ))}
          </select>
        </div>
      </header>

      <main className='mainclass'>
        <div className="chat-container">
          <div className="messages">
            {messages.map((msg, index) => (
              <Message
                key={index}
                text={msg.text}
                type={msg.type}
                id={msg.id}
                question={msg.text}
                audio={msg.audio}
                loading={msg.loading}
                onTextToSpeech={handleTextToSpeech}
              />
            ))}
            {loading && (
              <div className="loading">
                <ThreeDots color="#4CAF50" height={50} width={50} />
              </div>
            )}
            <div ref={messagesEndRef} /> {/* Ref for scrolling */}
          </div>
          {error && <p className="error">{error}</p>}
          <form onSubmit={handleFormSubmit} className="input-form">
            <input
              type="text"
              value={question}
              name="question"
              onChange={handleInputChange}
              placeholder="Type your question here..."
              required
            />
            <button type="submit" disabled={loading}>
              {loading ? 'Loading...' : 'Send'}
            </button>
          </form>
        </div>
      </main>
    </div>
  );
}

export default App;
